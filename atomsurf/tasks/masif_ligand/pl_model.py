import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import copy

import torch

# project
from atomsurf.tasks.masif_ligand.model import MasifLigandNet
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import multi_class_eval


class MasifLigandModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model = MasifLigandNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)
        #########################################################
        # Debugging configuration for loss spike investigation
        self.debug_enabled = False
        self.debug_start_step = 6294
        self.debug_end_step = 6304
        self.debug_dir = Path("/root/atomsurf/debug_spike_step_6299")
        
        # Storage for hooks
        self.debug_activations = {}
        self.debug_gradients = {}
        self.debug_hooks = []
        
        # Initialize debug directories
        if self.debug_enabled:
            self._init_debug_dirs()
        #########################################################

    def step(self, batch):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None
        labels = batch.label
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return loss, outputs, labels

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        accuracy_macro, accuracy_micro, accuracy_balanced, \
            precision_macro, precision_micro, \
            recall_macro, recall_micro, \
            f1_macro, f1_micro, \
            auroc_macro = multi_class_eval(logits, labels, K=7)
        self.log_dict({f"accuracy_balanced/{prefix}": accuracy_balanced,
                       f"precision_micro/{prefix}": precision_micro,
                       f"recall_micro/{prefix}": recall_micro,
                       f"f1_micro/{prefix}": f1_micro,
                       f"auroc_macro/{prefix}": auroc_macro,
                       }, on_epoch=True, batch_size=len(logits))

    #########################################################
    
    def _init_debug_dirs(self):
        """Initialize debug output directories."""
        self.debug_dir.mkdir(exist_ok=True, parents=True)
        (self.debug_dir / "batches").mkdir(exist_ok=True)
        (self.debug_dir / "model_states").mkdir(exist_ok=True)
        (self.debug_dir / "activations").mkdir(exist_ok=True)
        (self.debug_dir / "gradients").mkdir(exist_ok=True)
        (self.debug_dir / "metadata").mkdir(exist_ok=True)
        print(f"Debug directories initialized at {self.debug_dir}")
    
    def _register_debug_hooks(self):
        """Register forward and backward hooks on all modules."""
        self.debug_activations = {}
        self.debug_gradients = {}
        self.debug_hooks = []
        
        def make_forward_hook(name):
            def hook(module, input, output):
                # Store activation statistics and values
                if isinstance(output, torch.Tensor):
                    self.debug_activations[name] = {
                        'output': output.detach().cpu(),
                        'shape': list(output.shape),
                        'mean': output.detach().mean().item(),
                        'std': output.detach().std().item(),
                        'min': output.detach().min().item(),
                        'max': output.detach().max().item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item(),
                    }
                elif isinstance(output, tuple):
                    # Handle tuple outputs (e.g., from encoder blocks)
                    for i, out in enumerate(output):
                        if out is not None and isinstance(out, torch.Tensor):
                            self.debug_activations[f"{name}_output_{i}"] = {
                                'output': out.detach().cpu(),
                                'shape': list(out.shape),
                                'mean': out.detach().mean().item(),
                                'std': out.detach().std().item(),
                                'min': out.detach().min().item(),
                                'max': out.detach().max().item(),
                                'has_nan': torch.isnan(out).any().item(),
                                'has_inf': torch.isinf(out).any().item(),
                            }
                        elif out is not None and hasattr(out, 'x'):
                            # Handle graph/surface objects with .x attribute
                            x = out.x
                            self.debug_activations[f"{name}_output_{i}"] = {
                                'output': x.detach().cpu(),
                                'shape': list(x.shape),
                                'mean': x.detach().mean().item(),
                                'std': x.detach().std().item(),
                                'min': x.detach().min().item(),
                                'max': x.detach().max().item(),
                                'has_nan': torch.isnan(x).any().item(),
                                'has_inf': torch.isinf(x).any().item(),
                            }
            return hook
        
        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                # Store gradient statistics
                if grad_output[0] is not None:
                    grad = grad_output[0]
                    self.debug_gradients[name] = {
                        'gradient': grad.detach().cpu(),
                        'shape': list(grad.shape),
                        'mean': grad.detach().mean().item(),
                        'std': grad.detach().std().item(),
                        'min': grad.detach().min().item(),
                        'max': grad.detach().max().item(),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item(),
                        'norm': grad.detach().norm().item(),
                    }
            return hook
        
        # Register hooks on all named modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                h1 = module.register_forward_hook(make_forward_hook(name))
                h2 = module.register_full_backward_hook(make_backward_hook(name))
                self.debug_hooks.append(h1)
                self.debug_hooks.append(h2)
    
    def _remove_debug_hooks(self):
        """Remove all registered hooks."""
        for hook in self.debug_hooks:
            hook.remove()
        self.debug_hooks = []
    
    def _save_batch_data(self, batch, step):
        """Save batch data to disk."""
        batch_path = self.debug_dir / "batches" / f"batch_step_{step:05d}.pt"
        
        # Create a serializable version of the batch
        batch_data = {
            'label': batch.label.cpu(),
            'lig_coord': [lc.cpu() for lc in batch.lig_coord],
            'num_graphs': batch.num_graphs,
            'pocket_names': batch.pocket_name if hasattr(batch, 'pocket_name') else None,
        }
        
        # Save surface data if present
        if hasattr(batch, 'surface') and batch.surface is not None:
            surface_data = []
            for surf in batch.surface.to_data_list():
                surface_data.append({
                    'verts': surf.verts.cpu() if hasattr(surf, 'verts') else None,
                    'x': surf.x.cpu() if hasattr(surf, 'x') else None,
                    'faces': surf.faces.cpu() if hasattr(surf, 'faces') else None,
                })
            batch_data['surface'] = surface_data
        
        # Save graph data if present
        if hasattr(batch, 'graph') and batch.graph is not None:
            graph_data = []
            for g in batch.graph.to_data_list():
                graph_data.append({
                    'node_pos': g.node_pos.cpu() if hasattr(g, 'node_pos') else None,
                    'x': g.x.cpu() if hasattr(g, 'x') else None,
                    'edge_index': g.edge_index.cpu() if hasattr(g, 'edge_index') else None,
                })
            batch_data['graph'] = graph_data
        
        torch.save(batch_data, batch_path)
        print(f"Saved batch data to {batch_path}")
    
    def _save_model_state(self, step):
        """Save model state dict."""
        model_path = self.debug_dir / "model_states" / f"model_step_{step:05d}.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model state to {model_path}")
    
    def _save_activations(self, step):
        """Save activation data."""
        act_path = self.debug_dir / "activations" / f"activations_step_{step:05d}.pt"
        torch.save(self.debug_activations, act_path)
        print(f"Saved {len(self.debug_activations)} activations to {act_path}")
    
    def _save_gradients(self, step):
        """Save gradient data."""
        grad_path = self.debug_dir / "gradients" / f"gradients_step_{step:05d}.pt"
        torch.save(self.debug_gradients, grad_path)
        print(f"Saved {len(self.debug_gradients)} gradients to {grad_path}")
    
    def _save_metadata(self, step, loss, outputs, labels):
        """Save metadata about the step."""
        metadata = {
            'step': step,
            'loss': loss.item(),
            'output_shape': list(outputs.shape),
            'output_mean': outputs.detach().mean().item(),
            'output_std': outputs.detach().std().item(),
            'output_min': outputs.detach().min().item(),
            'output_max': outputs.detach().max().item(),
            'output_has_nan': torch.isnan(outputs).any().item(),
            'output_has_inf': torch.isinf(outputs).any().item(),
            'labels': labels.cpu().tolist(),
            'num_samples': len(labels),
        }
        
        # Add activation summary
        metadata['activation_summary'] = {}
        for name, act_info in self.debug_activations.items():
            metadata['activation_summary'][name] = {
                'shape': act_info['shape'],
                'mean': act_info['mean'],
                'std': act_info['std'],
                'min': act_info['min'],
                'max': act_info['max'],
                'has_nan': act_info['has_nan'],
                'has_inf': act_info['has_inf'],
            }
        
        # Add gradient summary
        metadata['gradient_summary'] = {}
        for name, grad_info in self.debug_gradients.items():
            metadata['gradient_summary'][name] = {
                'shape': grad_info['shape'],
                'mean': grad_info['mean'],
                'std': grad_info['std'],
                'min': grad_info['min'],
                'max': grad_info['max'],
                'has_nan': grad_info['has_nan'],
                'has_inf': grad_info['has_inf'],
                'norm': grad_info['norm'],
            }
        
        meta_path = self.debug_dir / "metadata" / f"metadata_step_{step:05d}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")
    
    def training_step(self, batch, batch_idx):
        """Override training_step to add debugging for specific steps."""
        # Get global step (cumulative across all epochs)
        current_step = self.global_step
        
        # Check if we need to debug this step
        if self.debug_enabled and self.debug_start_step <= current_step <= self.debug_end_step:
            print(f"\n{'='*80}")
            print(f"DEBUG: Starting global step {current_step}")
            print(f"{'='*80}")
            
            # Register hooks
            self._register_debug_hooks()
            
            # Save batch data before forward pass
            self._save_batch_data(batch, current_step)
            
            # Save model state before forward pass
            self._save_model_state(current_step)
        
        # Regular training step
        loss, logits, labels = self.step(batch)
        if loss is None:
            if self.debug_enabled and self.debug_start_step <= current_step <= self.debug_end_step:
                print(f"DEBUG: Step {current_step} - loss is None, skipping")
                self._remove_debug_hooks()
            return None
        
        # Log as usual
        self.log_dict({"loss/train": loss.item(),
                       "batch_size/train": len(logits)},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        self.train_res.append((logits.detach().cpu(), labels.detach().cpu()))
        
        # If debugging this step, save additional information after backward pass
        if self.debug_enabled and self.debug_start_step <= current_step <= self.debug_end_step:
            # Save activations (captured during forward pass)
            self._save_activations(current_step)
            
            # Save metadata
            self._save_metadata(current_step, loss, logits, labels)
            
            print(f"DEBUG: Step {current_step} completed - Loss: {loss.item():.6f}")
            print(f"{'='*80}\n")
        
        return loss
    
    def on_after_backward(self):
        """Called after backward pass - save gradients if debugging."""
        super().on_after_backward()
        
        # Get current global step
        current_step = self.global_step
        
        if self.debug_enabled and self.debug_start_step <= current_step <= self.debug_end_step:
            # Save gradients (captured during backward pass)
            self._save_gradients(current_step)
            
            # Remove hooks after saving
            self._remove_debug_hooks()
    #########################################################