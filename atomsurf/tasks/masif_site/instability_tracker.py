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
    
    @staticmethod
    def z_score_instability(losses):
        losses = np.array(losses)
        deltas = np.abs(np.diff(losses))  # |l_i - l_{i-1}|
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        if std_delta == 0:
            return 0.0  # perfectly smooth training

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
        if len(self.batches_per_epoch) < 2:
            return {"error": "Insufficient epochs for analysis"}
            
        # Calculate CV for batches WITHIN each epoch
        epoch_batch_cvs = []
        for epoch_batches in self.batches_per_epoch:
            if len(epoch_batches) > 1:  # Need at least 2 batches to calculate CV
                batch_mean = np.mean(epoch_batches)
                batch_std = np.std(epoch_batches)
                if batch_mean > 0:
                    cv = batch_std / batch_mean
                    epoch_batch_cvs.append(cv)
        
        # Calculate mean CV across epochs
        mean_cv = np.mean(epoch_batch_cvs) if epoch_batch_cvs else 0
        
        # Calculate z-score instability using epoch losses
        z_instability = self.z_score_instability(self.epoch_losses) if len(self.epoch_losses) > 1 else 0
        
        return {
            'mean_cv': mean_cv,
            'z_score_instability': z_instability,
            'total_epochs': len(self.batches_per_epoch)
        }
    
    def log_final_experiment(self, experiment_name, config, final_train_loss, 
                           final_val_loss=None, final_train_acc=None, 
                           final_val_acc=None, final_test_loss=None,
                           final_test_acc=None, comment=""):
        """Log final experiment results to CSV"""
        
        # Calculate metrics
        metrics = self.calculate_final_instability()
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'command': self.command,
            'total_epochs': metrics.get('total_epochs', 0),
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_test_loss': final_test_loss,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'final_test_acc': final_test_acc,
            'mean_cv': metrics.get('mean_cv', 0),
            'z_score_instability': metrics.get('z_score_instability', 0),
            'comment': comment
        }
        
        # Add config info
        if hasattr(config, 'optimizer'):
            log_entry['lr'] = getattr(config.optimizer, 'lr', 'unknown')
            log_entry['weight_decay'] = getattr(config.optimizer, 'weight_decay', 'unknown')
        if hasattr(config, 'loader'):
            log_entry['batch_size'] = getattr(config.loader, 'batch_size', 'unknown')
        if hasattr(config, 'epochs'):
            log_entry['num_epochs'] = getattr(config, 'epochs', 'unknown')
        
        # Save to CSV
        df = pd.DataFrame([log_entry])
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_path, mode='w', header=True, index=False)
            
        return log_entry 