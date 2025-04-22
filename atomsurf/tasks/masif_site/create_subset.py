import os
import argparse
import torch
from pathlib import Path
import shutil
import sys

sys.path.append(os.path.abspath('atomsurf'))
from atomsurf.protein import *  # This imports the necessary classes for loading the data

def get_protein_stats(protein_id, data_dir):
    """Get the graph statistics of a protein"""
    graph_file = os.path.join(data_dir, 'rgraph', f"{protein_id}.pt")
    try:
        data = torch.load(graph_file)
        if hasattr(data, 'num_nodes'):
            stats = {
                'num_nodes': data.num_nodes,
                'num_edges': data.edge_index.shape[1],
                'density': None,
                'avg_degree': None
            }
            if data.num_nodes > 1:
                stats['density'] = (2 * data.edge_index.shape[1]) / (data.num_nodes * (data.num_nodes - 1))
                stats['avg_degree'] = data.edge_index.shape[1] / data.num_nodes
            return stats
    except Exception as e:
        print(f"Error loading {graph_file}: {str(e)}")
    return None

def create_filtered_subset(data_dir, output_dir, min_size=120, max_size=320, 
                         min_density=0.05, max_density=0.5, 
                         min_degree=10, max_degree=40):
    """Create a subset of proteins filtered by graph properties"""
    splits_dir = os.path.join(data_dir, 'splits')
    
    # Create output directory and its splits subdirectory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'splits'), exist_ok=True)
    
    # Read all proteins
    with open(os.path.join(splits_dir, 'train_list.txt'), 'r') as f:
        train_proteins = [line.strip() for line in f]
    with open(os.path.join(splits_dir, 'test_list.txt'), 'r') as f:
        test_proteins = [line.strip() for line in f]
        
    filtered_train = []
    filtered_test = []
    
    # Filter train proteins
    print("Filtering training proteins...")
    excluded_train = {'size': 0, 'density': 0, 'degree': 0}
    for protein in train_proteins:
        stats = get_protein_stats(protein, data_dir)
        if stats:
            if not (min_size <= stats['num_nodes'] <= max_size):
                excluded_train['size'] += 1
                continue
            if stats['density'] is None or not (min_density <= stats['density'] <= max_density):
                excluded_train['density'] += 1
                continue
            if stats['avg_degree'] is None or not (min_degree <= stats['avg_degree'] <= max_degree):
                excluded_train['degree'] += 1
                continue
            filtered_train.append(protein)
    
    # Filter test proteins
    print("Filtering test proteins...")
    excluded_test = {'size': 0, 'density': 0, 'degree': 0}
    for protein in test_proteins:
        stats = get_protein_stats(protein, data_dir)
        if stats:
            if not (min_size <= stats['num_nodes'] <= max_size):
                excluded_test['size'] += 1
                continue
            if stats['density'] is None or not (min_density <= stats['density'] <= max_density):
                excluded_test['density'] += 1
                continue
            if stats['avg_degree'] is None or not (min_degree <= stats['avg_degree'] <= max_degree):
                excluded_test['degree'] += 1
                continue
            filtered_test.append(protein)
    
    # Save new splits
    with open(os.path.join(output_dir, 'splits', 'train_list.txt'), 'w') as f:
        f.write('\n'.join(filtered_train))
    with open(os.path.join(output_dir, 'splits', 'test_list.txt'), 'w') as f:
        f.write('\n'.join(filtered_test))
    
    # Print statistics
    print(f"\nOriginal dataset size: {len(train_proteins)} train, {len(test_proteins)} test")
    print(f"Filtered dataset size: {len(filtered_train)} train, {len(filtered_test)} test")
    print("\nExcluded proteins from train set:")
    print(f"  Due to size: {excluded_train['size']}")
    print(f"  Due to density: {excluded_train['density']}")
    print(f"  Due to degree: {excluded_train['degree']}")
    print("\nExcluded proteins from test set:")
    print(f"  Due to size: {excluded_test['size']}")
    print(f"  Due to density: {excluded_test['density']}")
    print(f"  Due to degree: {excluded_test['degree']}")
    
    return filtered_train, filtered_test

def main():
    parser = argparse.ArgumentParser(description='Create a filtered subset of the MaSIF-site dataset based on graph properties')
    parser.add_argument('--data_dir', type=str, default='/home/tamara/data/masif_site',
                        help='Path to the original dataset')
    parser.add_argument('--output_dir', type=str, default='/home/tamara/data/masif_site_filtered',
                        help='Path to save the filtered dataset')
    parser.add_argument('--min_size', type=int, default=120,
                        help='Minimum graph size (number of nodes)')
    parser.add_argument('--max_size', type=int, default=320,
                        help='Maximum graph size (number of nodes)')
    parser.add_argument('--min_density', type=float, default=0.05,
                        help='Minimum graph density')
    parser.add_argument('--max_density', type=float, default=0.5,
                        help='Maximum graph density')
    parser.add_argument('--min_degree', type=float, default=10,
                        help='Minimum average node degree')
    parser.add_argument('--max_degree', type=float, default=40,
                        help='Maximum average node degree')
    
    args = parser.parse_args()
    
    filtered_train, filtered_test = create_filtered_subset(
        args.data_dir,
        args.output_dir,
        args.min_size,
        args.max_size,
        args.min_density,
        args.max_density,
        args.min_degree,
        args.max_degree
    )

if __name__ == '__main__':
    main() 