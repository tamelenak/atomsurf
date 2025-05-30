import torch
from torch.utils.data import Sampler
import numpy as np
from typing import List, Dict, Optional
import os

class UniformBatchSampler(Sampler):
    """
    Sampler that creates batches with similarly-sized proteins.
    Proteins are sorted by size and binned, then batches are created from within these bins.
    """
    
    def __init__(self, 
                 dataset,
                 batch_size: int,
                 num_bins: int = 5,
                 shuffle: bool = True,
                 drop_last: bool = False):
        """
        Args:
            dataset: The MasifSiteDataset instance
            batch_size: Number of proteins per batch
            num_bins: Number of size bins to create
            shuffle: Whether to shuffle proteins within bins
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get size of each protein by loading the first graph
        self.protein_sizes = {}
        for idx in range(len(dataset)):
            try:
                data = dataset[idx]
                if data is not None and hasattr(data.graph, 'num_nodes'):
                    self.protein_sizes[idx] = data.graph.num_nodes
            except Exception as e:
                print(f"Error loading protein at index {idx}: {str(e)}")
        
        # Sort proteins by size
        sorted_proteins = sorted(self.protein_sizes.items(), key=lambda x: x[1])
        
        # Create bins of proteins with similar sizes
        self.bins = []
        proteins_per_bin = len(sorted_proteins) // num_bins
        for i in range(num_bins):
            start_idx = i * proteins_per_bin
            end_idx = start_idx + proteins_per_bin if i < num_bins - 1 else len(sorted_proteins)
            bin_proteins = [p[0] for p in sorted_proteins[start_idx:end_idx]]
            self.bins.append(bin_proteins)
        
        # Calculate number of batches
        self.num_batches = sum(len(bin_) // batch_size for bin_ in self.bins)
        
        print("\nBatch size statistics:")
        for i, bin_ in enumerate(self.bins):
            if bin_:
                sizes = [self.protein_sizes[p] for p in bin_]
                print(f"Bin {i}: {len(bin_)} proteins, size range: {min(sizes)}-{max(sizes)} nodes")
    
    def __iter__(self):
        # Shuffle proteins within each bin if requested
        if self.shuffle:
            for bin_ in self.bins:
                np.random.shuffle(bin_)
        
        # Create batches from each bin
        all_batches = []
        for bin_ in self.bins:
            num_proteins = len(bin_)
            num_complete_batches = num_proteins // self.batch_size
            
            for i in range(num_complete_batches):
                start_idx = i * self.batch_size
                batch = bin_[start_idx:start_idx + self.batch_size]
                all_batches.append(batch)
        
        # Shuffle the order of batches if requested
        if self.shuffle:
            np.random.shuffle(all_batches)
        
        # Flatten batches into a list of indices
        return iter([idx for batch in all_batches for idx in batch])
    
    def __len__(self):
        return self.num_batches * self.batch_size

# Example usage in your training script:
"""
from uniform_batch_sampler import UniformBatchSampler

# Create the sampler
sampler = UniformBatchSampler(
    dataset=your_dataset,
    batch_size=32,
    num_bins=5,  # Adjust based on your dataset size
    shuffle=True
)

# Use it in your DataLoader
train_loader = DataLoader(
    dataset=your_dataset,
    batch_sampler=sampler,  # Use batch_sampler instead of batch_size
    num_workers=4
)
""" 