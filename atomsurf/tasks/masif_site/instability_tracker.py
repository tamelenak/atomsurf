import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys

class InstabilityTracker:
    """Track training stability - focus on batch variation within epochs"""
    
    def __init__(self, csv_path="experiment_tracker.csv"):
        self.csv_path = csv_path
        
        # Track data per epoch
        self.batches_per_epoch = []  # List of lists: [[epoch0_batches], [epoch1_batches], ...]
        self.current_epoch_batches = []  # Current epoch's batch losses
        self.epoch_losses = []  # Track average loss per epoch
        
        # Store command line arguments
        self.command = " ".join(sys.argv)
        
        # Log experiment start immediately
        self.log_experiment_start()
    
    def log_experiment_start(self):
        """Log experiment start to CSV"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': 'unknown',  # Will be updated later
            'command': self.command,
            'train_auroc': None,
            'val_auroc': None,
            'test_auroc': None,
            'lr': 'unknown',
            'weight_decay': 'unknown',
            'batch_size': 'unknown',
            'num_epochs': 'unknown',
            'encoder_name': 'unknown'
        }
        
        # Save to CSV
        df = pd.DataFrame([log_entry])
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_path, mode='w', header=True, index=False)
    
    def update_experiment_info(self, experiment_name, config):
        """Update experiment information in the CSV"""
        # Read existing CSV
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            # Find the last row (most recent entry) and update it
            if len(df) > 0:
                last_idx = len(df) - 1
                
                # Update experiment name
                df.loc[last_idx, 'experiment_name'] = experiment_name
                
                # Add config info
                if hasattr(config, 'optimizer'):
                    df.loc[last_idx, 'lr'] = getattr(config.optimizer, 'lr', 'unknown')
                    df.loc[last_idx, 'weight_decay'] = getattr(config.optimizer, 'weight_decay', 'unknown')
                if hasattr(config, 'loader'):
                    df.loc[last_idx, 'batch_size'] = getattr(config.loader, 'batch_size', 'unknown')
                if hasattr(config, 'epochs'):
                    df.loc[last_idx, 'num_epochs'] = getattr(config, 'epochs', 'unknown')
                if hasattr(config, 'encoder'):
                    df.loc[last_idx, 'encoder_name'] = getattr(config.encoder, 'name', 'unknown')
                
                # Save updated CSV
                df.to_csv(self.csv_path, index=False)
    
    def update_final_metrics(self, train_auroc=None, val_auroc=None, test_auroc=None):
        """Update final AUROC metrics in the CSV"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            # Find the last row and update AUROC metrics
            if len(df) > 0:
                last_idx = len(df) - 1
                
                if train_auroc is not None:
                    df.loc[last_idx, 'train_auroc'] = train_auroc
                if val_auroc is not None:
                    df.loc[last_idx, 'val_auroc'] = val_auroc
                if test_auroc is not None:
                    df.loc[last_idx, 'test_auroc'] = test_auroc
                
                # Save updated CSV
                df.to_csv(self.csv_path, index=False)

    @staticmethod
    def z_score_instability(losses):
        losses = np.array(losses)
        deltas = np.abs(np.diff(losses))
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        z_scores = (deltas - mean_delta) / std_delta
        return np.mean(np.abs(z_scores))
        
    def update_batch(self, loss):
        """Store batch loss for current epoch"""
        self.current_epoch_batches.append(loss)
    
    def update_epoch(self, epoch_loss=None):
        """Finalize current epoch's batches"""
        # Save current epoch's batches and start new epoch
        if self.current_epoch_batches:
            self.batches_per_epoch.append(self.current_epoch_batches.copy())
            if epoch_loss is not None:
                self.epoch_losses.append(epoch_loss)
        self.current_epoch_batches = []  # Reset for next epoch
    
    def calculate_final_instability(self):
        """Calculate instability metrics using per-epoch batch data"""
            
        # Calculate CV for batches WITHIN each epoch
        epoch_batch_cvs = []
        epoch_batch_zscores = []
        for epoch_batches in self.batches_per_epoch:
            if len(epoch_batches) > 1:  # Need at least 2 batches to calculate CV
                batch_mean = np.mean(epoch_batches)
                batch_std = np.std(epoch_batches)
                if batch_mean > 0:
                    cv = batch_std / batch_mean
                    epoch_batch_cvs.append(cv)
                # batch-level z-score instability
                deltas = np.abs(np.diff(epoch_batches))
                mean_delta = np.mean(deltas)
                std_delta = np.std(deltas)
                if std_delta == 0:
                    z_score = 0.0
                else:
                    z_scores = (deltas - mean_delta) / std_delta
                    z_score = np.mean(np.abs(z_scores))
                epoch_batch_zscores.append(z_score)
        
        # Calculate mean CV across epochs
        mean_cv = np.mean(epoch_batch_cvs) if epoch_batch_cvs else 0
        
        # Calculate mean batch-level z-score instability
        mean_batch_zscore = np.mean(epoch_batch_zscores) if epoch_batch_zscores else 0
        
        # Calculate z-score instability using epoch losses
        z_instability = self.z_score_instability(self.epoch_losses) if len(self.epoch_losses) > 1 else 0
        
        return {
            'mean_cv': mean_cv,
            'z_score_instability': z_instability,
            'mean_batch_zscore_instability': mean_batch_zscore,
            'total_epochs': len(self.batches_per_epoch)
        }
    
    def log_final_experiment(self, experiment_name, config, final_train_loss, 
                           final_val_loss=None, final_train_acc=None, 
                           final_val_acc=None, final_test_loss=None,
                           final_test_acc=None, comment=""):
        """Log final experiment results to CSV - DEPRECATED, use update_final_metrics instead"""
        # This method is kept for backward compatibility but is deprecated
        # The new approach logs at start and updates metrics at end
        pass 