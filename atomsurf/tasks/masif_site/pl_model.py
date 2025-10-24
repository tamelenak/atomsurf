import os
import sys

import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_site.model import MasifSiteNet
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import compute_accuracy, compute_auroc
from atomsurf.tasks.masif_site.instability_tracker import InstabilityTracker


def masif_site_loss(preds, labels, use_weighted_bce: bool = False):
    """Compute balanced BCE loss.

    If *use_weighted_bce* is True the whole batch is used and the minority
    class is up-weighted via *pos_weight* so no stochastic down-sampling is
    involved (lower gradient variance).  Otherwise the original random
    down-sampling strategy is applied for backward compatibility.
    Returns (loss, preds_used, labels_used) or (None, None, None) if the batch
    contains only one class.
    """

    # Handle edge case: single-class batch
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        print("[Warning] Only one class present in batch. Skipping loss computation.")
        return None, None, None

    if use_weighted_bce:
        # Compute class weights: pos gets the inverse frequency of pos/neg ratio
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        pos_weight = n_neg.float() / n_pos.float()

        loss = F.binary_cross_entropy_with_logits(
            preds,
            labels.float(),
            pos_weight=pos_weight,
        )
        return loss, preds, labels

    # -------- Original stochastic subsampling version --------
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

    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
    return loss, preds_concat, labels_concat


class MasifSiteModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MasifSiteNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)
        
        # Initialize instability tracker
        self.instability_tracker = InstabilityTracker(
            csv_path="experiment_tracker.csv"
        )
        
        # Store config for logging
        self.cfg = cfg
        
        # Update experiment info after initialization
        self.instability_tracker.update_experiment_info(self.cfg.run_name, self.cfg)

    def step(self, batch):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None, None
        labels = torch.concatenate(batch.label)
        out_surface_batch = self(batch)
        outputs = out_surface_batch.x.flatten()
        
        # Use weighted BCE if requested in config
        loss, outputs_concat, labels_concat = masif_site_loss(
            outputs, labels, use_weighted_bce=self.hparams.cfg.loss.weighted_bce
        )
            
        if loss is None:
            return None, None, None, None

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
        """Update final AUROC metrics when training ends"""
        # Get AUROC metrics from the trainer callback metrics
        train_auroc = self.trainer.callback_metrics.get('auroc/train', None)
        val_auroc = self.trainer.callback_metrics.get('auroc/val', None)
        test_auroc = self.trainer.callback_metrics.get('auroc/test', None)
        
        # Convert to float if they are tensors
        train_auroc = train_auroc.item() if train_auroc is not None and hasattr(train_auroc, 'item') else None
        val_auroc = val_auroc.item() if val_auroc is not None and hasattr(val_auroc, 'item') else None
        test_auroc = test_auroc.item() if test_auroc is not None and hasattr(test_auroc, 'item') else None
        
        # Update final metrics in the CSV
        self.instability_tracker.update_final_metrics(
            train_auroc=train_auroc,
            val_auroc=val_auroc,
            test_auroc=test_auroc
        )

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        # Check if both classes are present
        unique_labels = torch.unique(labels)
        if unique_labels.numel() < 2:
            print(f"[Warning] Only one class present in validation labels for prefix '{prefix}'. Setting auroc to 0.5.")
            auroc = 0.5
        else:
            auroc = compute_auroc(predictions=logits, labels=labels)
        acc = compute_accuracy(predictions=logits, labels=labels, add_sigmoid=True)
        self.log_dict({
            f"auroc/{prefix}": auroc,
            f"acc/{prefix}": acc
        }, on_epoch=True, batch_size=len(logits))
