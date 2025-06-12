import os
import sys

import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_site.model_graph_only_wrapper import GraphOnlyMasifSiteWrapper
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import compute_accuracy, compute_auroc
from atomsurf.tasks.masif_site.focal_loss import masif_site_focal_loss, weighted_masif_site_loss
from atomsurf.tasks.masif_site.instability_tracker import InstabilityTracker


def masif_site_loss(preds, labels):
    # Same loss function as the original
    pos_preds = preds[labels == 1]
    neg_preds = preds[labels == 0]
    pos_labels = torch.ones_like(pos_preds)
    neg_labels = torch.zeros_like(neg_preds)

    # Subsample majority class to get balanced loss
    n_points_sample = min(len(pos_labels), len(neg_labels))
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]
    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]

    # Compute loss on these prediction/GT pairs
    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
    return loss, preds_concat, labels_concat


class GraphOnlyMasifSiteModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Use the wrapper around the standard MasifSiteNet
        self.model = GraphOnlyMasifSiteWrapper(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)
        
        # Configure loss function
        self.loss_type = getattr(cfg, 'loss_type', 'default')
        self.focal_alpha = getattr(cfg, 'focal_alpha', 0.25)
        self.focal_gamma = getattr(cfg, 'focal_gamma', 2.0)
        
        # Initialize instability tracker
        self.instability_tracker = InstabilityTracker(
            csv_path="experiment_tracker_graph_only.csv"
        )
        
        # Store config for logging
        self.cfg = cfg

    def step(self, batch):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None, None
            
        # Get labels from surface (same as original)
        labels = torch.concatenate(batch.label)
        
        # Forward pass through graph-only wrapper
        out_surface_batch = self(batch)
        outputs = out_surface_batch.x.flatten()
        
        # Use the same loss functions as original
        if self.loss_type == 'focal':
            loss, outputs_concat, labels_concat = masif_site_focal_loss(
                outputs, labels, alpha=self.focal_alpha, gamma=self.focal_gamma
            )
        elif self.loss_type == 'weighted':
            loss, outputs_concat, labels_concat = weighted_masif_site_loss(outputs, labels)
        else:
            loss, outputs_concat, labels_concat = masif_site_loss(outputs, labels)
            
        accuracy = compute_accuracy(predictions=outputs_concat, labels=labels_concat, add_sigmoid=True)
        
        # Log batch statistics
        if self.training:
            if hasattr(batch.graph, 'node_len'):
                self.log_dict({
                    "size/nodes": batch.graph.node_len.float().sum(),
                    "size/points": len(labels),
                    "size/loss": loss.item()
                }, on_epoch=True, batch_size=len(labels))
        return loss, outputs_concat, labels_concat, accuracy

    def training_step(self, batch, batch_idx):
        loss, logits, labels, accuracy = self.step(batch)
        if loss is None:
            return None
            
        # Store batch loss for current epoch analysis
        self.instability_tracker.update_batch(loss.item())
        
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        self.train_res.append((logits.detach().cpu(), labels.detach().cpu()))
        return {"loss": loss, "accuracy": accuracy}

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        
        # Store epoch loss and finalize current epoch's batches
        train_loss = self.trainer.callback_metrics.get('loss/train_epoch', 0)
        if hasattr(train_loss, 'item'):
            self.instability_tracker.update_epoch(train_loss.item())

    def on_fit_end(self):
        """Calculate and log final instability metrics when training ends"""
        # Calculate final instability metrics
        final_metrics = self.instability_tracker.calculate_final_instability()
        
        # Get final performance metrics
        final_train_loss = self.trainer.callback_metrics.get('loss/train_epoch', None)
        final_val_loss = self.trainer.callback_metrics.get('loss/val', None)
        final_test_loss = self.trainer.callback_metrics.get('loss/test', None)
        
        # Get accuracy metrics from the metrics dictionary
        final_train_acc = self.trainer.callback_metrics.get('acc/train', None)
        final_val_acc = self.trainer.callback_metrics.get('acc/val', None)
        final_test_acc = self.trainer.callback_metrics.get('acc/test', None)
        
        # Log final experiment summary
        self.instability_tracker.log_final_experiment(
            experiment_name=f"{self.cfg.run_name}_graph_only",
            config=self.cfg,
            final_train_loss=final_train_loss.item() if final_train_loss is not None and hasattr(final_train_loss, 'item') else 0,
            final_val_loss=final_val_loss.item() if final_val_loss is not None and hasattr(final_val_loss, 'item') else None,
            final_test_loss=final_test_loss.item() if final_test_loss is not None and hasattr(final_test_loss, 'item') else None,
            final_train_acc=final_train_acc.item() if final_train_acc is not None and hasattr(final_train_acc, 'item') else None,
            final_val_acc=final_val_acc.item() if final_val_acc is not None and hasattr(final_val_acc, 'item') else None,
            final_test_acc=final_test_acc.item() if final_test_acc is not None and hasattr(final_test_acc, 'item') else None,
            comment=f"{self.cfg.comment} - Graph Only Comparison"
        )

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        auroc = compute_auroc(predictions=logits, labels=labels)
        acc = compute_accuracy(predictions=logits, labels=labels, add_sigmoid=True)
        self.log_dict({
            f"auroc/{prefix}": auroc,
            f"acc/{prefix}": acc
        }, on_epoch=True, batch_size=len(logits)) 