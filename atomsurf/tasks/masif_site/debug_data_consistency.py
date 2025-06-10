import torch
from atomsurf.tasks.masif_site.data_loader import MasifSiteDataModule
import hydra
from omegaconf import DictConfig
import pandas as pd
from collections import defaultdict
import os
import sys

def check_data_consistency(cfg, num_epochs=3):
    """Check data consistency across multiple epochs"""
    
    # Get data module and loaders
    dm = MasifSiteDataModule(cfg)
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    
    epoch_stats = []
    failed_loads = defaultdict(int)
    batch_sizes = []
    
    print(f"Checking data consistency across {num_epochs} epochs...")
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Expected batches per epoch: {len(train_loader)}")
    
    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch} ===")
        
        total_nodes = 0
        total_points = 0
        successful_batches = 0
        failed_batches = 0
        nodes_per_batch = []
        points_per_batch = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                failed_batches += 1
                failed_loads[f'epoch_{epoch}_batch_{batch_idx}'] += 1
                continue
                
            # Check if batch would be skipped by model
            if batch.num_graphs < cfg.min_batch_size:
                print(f"  Batch {batch_idx}: SKIPPED (num_graphs={batch.num_graphs} < min_batch_size={cfg.min_batch_size})")
                failed_batches += 1
                continue
                
            # Count nodes and points
            if hasattr(batch.graph, 'node_len'):
                batch_nodes = batch.graph.node_len.float().sum().item()
                total_nodes += batch_nodes
                nodes_per_batch.append(batch_nodes)
            
            if hasattr(batch, 'label'):
                batch_points = len(torch.concatenate(batch.label))
                total_points += batch_points
                points_per_batch.append(batch_points)
            
            batch_sizes.append(batch.num_graphs)
            successful_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: nodes={batch_nodes:.0f}, points={batch_points}, graphs={batch.num_graphs}")
        
        epoch_stats.append({
            'epoch': epoch,
            'total_nodes': total_nodes,
            'total_points': total_points,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'avg_nodes_per_batch': sum(nodes_per_batch) / len(nodes_per_batch) if nodes_per_batch else 0,
            'avg_points_per_batch': sum(points_per_batch) / len(points_per_batch) if points_per_batch else 0,
            'unique_batch_sizes': len(set(batch_sizes[-successful_batches:])) if successful_batches > 0 else 0
        })
        
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total points: {total_points}")
        print(f"  Successful batches: {successful_batches}")
        print(f"  Failed batches: {failed_batches}")
    
    # Analysis
    print("\n=== SUMMARY ===")
    df = pd.DataFrame(epoch_stats)
    print(df)
    
    print(f"\nNode count variation:")
    print(f"  Min: {df['total_nodes'].min()}")
    print(f"  Max: {df['total_nodes'].max()}")
    print(f"  Std: {df['total_nodes'].std():.2f}")
    print(f"  CV: {df['total_nodes'].std() / df['total_nodes'].mean() * 100:.2f}%")
    
    if failed_loads:
        print(f"\nFailed loads detected: {sum(failed_loads.values())} total")
        for key, count in list(failed_loads.items())[:10]:  # Show first 10
            print(f"  {key}: {count}")
    
    # Check if it's a shuffling issue
    print(f"\nBatch size consistency:")
    print(f"  Unique batch sizes seen: {len(set(batch_sizes))}")
    print(f"  Batch sizes: {set(batch_sizes)}")
    
    return epoch_stats, failed_loads

def check_dataloader_determinism(cfg):
    """Check if the dataloader is deterministic"""
    print("\n=== CHECKING DATALOADER DETERMINISM ===")
    
    # Set seeds
    torch.manual_seed(42)
    dm = MasifSiteDataModule(cfg)
    dm.setup('fit')
    train_loader1 = dm.train_dataloader()
    
    torch.manual_seed(42) 
    dm = MasifSiteDataModule(cfg)
    dm.setup('fit')
    train_loader2 = dm.train_dataloader()
    
    # Get first few batches from each loader
    batches1 = []
    batches2 = []
    
    for i, (batch1, batch2) in enumerate(zip(train_loader1, train_loader2)):
        if i >= 5:  # Just check first 5 batches
            break
        if batch1 is not None and batch2 is not None:
            batches1.append(batch1.protein_name if hasattr(batch1, 'protein_name') else None)
            batches2.append(batch2.protein_name if hasattr(batch2, 'protein_name') else None)
    
    print(f"First 5 batches loader 1: {batches1}")
    print(f"First 5 batches loader 2: {batches2}")
    print(f"Identical: {batches1 == batches2}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== DATA CONSISTENCY DEBUG ===")
    
    # Check consistency
    epoch_stats, failed_loads = check_data_consistency(cfg, num_epochs=15)
    
    # Check determinism
    check_dataloader_determinism(cfg)
    
    # Save results
    df = pd.DataFrame(epoch_stats)
    df.to_csv('data_consistency_debug.csv', index=False)
    print(f"\nResults saved to data_consistency_debug.csv")

if __name__ == "__main__":
    main() 