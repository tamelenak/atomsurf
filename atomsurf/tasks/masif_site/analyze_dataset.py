import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath('atomsurf'))
from atomsurf.protein import *

def load_protein_data(protein_id, data_dir):
    """Load protein data and return relevant features"""
    stats = {}
    
    # Load surface data
    surface_file = os.path.join(data_dir, 'surfaces_0.1_False', f"{protein_id}.pt")
    if os.path.exists(surface_file):
        try:
            data = torch.load(surface_file)
            if hasattr(data, 'verts') and data.verts is not None:
                stats['surface_vertices'] = len(data.verts)
        except Exception as e:
            print(f"\nError loading surface file {surface_file}: {str(e)}")
    
    # Load graph data
    graph_file = os.path.join(data_dir, 'rgraph', f"{protein_id}.pt")
    if os.path.exists(graph_file):
        try:
            data = torch.load(graph_file)
            if hasattr(data, 'num_nodes'):
                stats['graph_nodes'] = data.num_nodes
                if hasattr(data, 'edge_index'):
                    stats['graph_edges'] = data.edge_index.shape[1]
                    if data.num_nodes > 1:
                        stats['graph_density'] = (2 * data.edge_index.shape[1]) / (data.num_nodes * (data.num_nodes - 1))
                        stats['avg_degree'] = data.edge_index.shape[1] / data.num_nodes
        except Exception as e:
            print(f"\nError loading graph file {graph_file}: {str(e)}")
    
    return stats

def plot_size_correlation(df, output_dir):
    """Plot correlation between surface and graph sizes"""
    plt.figure(figsize=(10, 10))
    plt.scatter(df['surface_vertices'], df['graph_nodes'], alpha=0.5)
    plt.xlabel('Surface Size (vertices)')
    plt.ylabel('Graph Size (nodes)')
    plt.title('Correlation between Surface and Graph Sizes')
    
    # Add correlation coefficient
    corr = df['surface_vertices'].corr(df['graph_nodes'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=plt.gca().transAxes)
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'size_correlation.png'))
    plt.close()

def analyze_dataset(data_dir, output_dir):
    """Analyze the dataset for outliers and statistics"""
    os.makedirs(output_dir, exist_ok=True)
    splits_dir = os.path.join(data_dir, 'splits_filtered')
    
    # Read protein lists
    with open(os.path.join(splits_dir, 'train_list.txt'), 'r') as f:
        train_proteins = [line.strip() for line in f]
    with open(os.path.join(splits_dir, 'test_list.txt'), 'r') as f:
        test_proteins = [line.strip() for line in f]
    
    all_proteins = train_proteins + test_proteins
    all_stats = []
    
    print(f"\nAnalyzing {len(all_proteins)} proteins...")
    
    # Collect statistics for all proteins
    for protein in tqdm(all_proteins):
        stats = load_protein_data(protein, data_dir)
        if stats:
            stats['protein_id'] = protein
            stats['split'] = 'train' if protein in train_proteins else 'test'
            all_stats.append(stats)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_stats)
    
    # Basic statistics for numerical columns
    base_cols = ['surface_vertices', 'graph_nodes', 'graph_edges', 'graph_density', 'avg_degree']
    numeric_cols = [col for col in base_cols if col in df.columns]
    
    if 'surface_faces' in df.columns:
        numeric_cols.append('surface_faces')
    
    print("\nDataset Statistics:")
    print("-" * 50)
    print(df[numeric_cols].describe())
    
    # Plot distributions
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, hue='split', multiple="stack")
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
            plt.close()
            
            # Box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='split', y=col)
            plt.title(f'Box Plot of {col} by Split')
            plt.savefig(os.path.join(output_dir, f'{col}_boxplot.png'))
            plt.close()
    
    # Plot surface-graph correlation
    if 'surface_vertices' in df.columns and 'graph_nodes' in df.columns:
        plot_size_correlation(df, output_dir)
    
    # Save detailed statistics
    df.to_csv(os.path.join(output_dir, 'dataset_statistics.csv'), index=False)
    
    # Print suggested filtering ranges
    for col in ['surface_vertices', 'graph_nodes']:
        if col in df.columns:
            percentiles = np.percentile(df[col], [25, 50, 75])
            print(f"\nSuggested {col} ranges:")
            print(f"Small: {df[col].min():.0f} - {percentiles[0]:.0f}")
            print(f"Medium: {percentiles[0]:.0f} - {percentiles[2]:.0f}")
            print(f"Large: {percentiles[2]:.0f} - {df[col].max():.0f}")
    
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze MaSIF-site dataset')
    parser.add_argument('--data_dir', type=str, default='/root/atomsurf/masif_site_data',
                        help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis',
                        help='Path to save analysis results')
    
    args = parser.parse_args()
    analyze_dataset(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main() 