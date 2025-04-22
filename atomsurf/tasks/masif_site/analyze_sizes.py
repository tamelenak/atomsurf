import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath('atomsurf'))
from atomsurf.protein import *  # This imports the necessary classes for loading the data

def get_sizes(protein_id, surface_dir, graph_dir):
    """Get both surface and graph sizes for a protein"""
    sizes = {'surface': None, 'graph': None}
    
    # Get surface size
    surface_file = os.path.join(surface_dir, f"{protein_id}.pt")
    if os.path.exists(surface_file):
        try:
            data = torch.load(surface_file)
            if hasattr(data, 'verts') and data.verts is not None:
                sizes['surface'] = len(data.verts)
        except Exception as e:
            print(f"\nError loading surface file {surface_file}: {str(e)}")
    
    # Get graph size
    graph_file = os.path.join(graph_dir, f"{protein_id}.pt")
    if os.path.exists(graph_file):
        try:
            data = torch.load(graph_file)
            if hasattr(data, 'num_nodes'):
                sizes['graph'] = data.num_nodes
        except Exception as e:
            print(f"\nError loading graph file {graph_file}: {str(e)}")
    
    return sizes

def plot_size_distribution(sizes, title, filename):
    """Create and save a histogram of sizes"""
    plt.figure(figsize=(12, 6))
    plt.hist(sizes, bins=50, edgecolor='black')
    plt.title(title)
    plt.xlabel('Number of Vertices/Nodes')
    plt.ylabel('Count')
    
    mean_val = np.mean(sizes)
    median_val = np.median(sizes)
    
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='g', linestyle='dashed', linewidth=2,
                label=f'Median: {median_val:.0f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_correlation(surface_sizes, graph_sizes, protein_ids):
    """Create a scatter plot showing correlation between surface and graph sizes"""
    plt.figure(figsize=(10, 10))
    plt.scatter(surface_sizes, graph_sizes, alpha=0.5)
    plt.xlabel('Surface Size (vertices)')
    plt.ylabel('Graph Size (nodes)')
    plt.title('Correlation between Surface and Graph Sizes')
    
    # Add correlation coefficient
    corr = np.corrcoef(surface_sizes, graph_sizes)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
             transform=plt.gca().transAxes)
    
    # Add grid and save
    plt.grid(True, alpha=0.3)
    plt.savefig('size_correlation.png')
    plt.close()
    
    # Find outliers and save to file
    mean_surface = np.mean(surface_sizes)
    std_surface = np.std(surface_sizes)
    mean_graph = np.mean(graph_sizes)
    std_graph = np.std(graph_sizes)
    
    outliers = []
    for i, (s, g, pid) in enumerate(zip(surface_sizes, graph_sizes, protein_ids)):
        if (abs(s - mean_surface) > 2*std_surface or 
            abs(g - mean_graph) > 2*std_graph):
            outliers.append(f"{pid}: surface={s}, graph={g}")
    
    if outliers:
        with open('size_outliers.txt', 'w') as f:
            f.write('\n'.join(outliers))

def analyze_sizes(data_dir):
    """Analyze both surface and graph sizes"""
    surface_dir = os.path.join(data_dir, 'surfaces_0.1_False')
    graph_dir = os.path.join(data_dir, 'rgraph')
    splits_dir = os.path.join(data_dir, 'splits')
    
    print(f"\nAnalyzing protein sizes...")
    print(f"Surface directory: {surface_dir}")
    print(f"Graph directory: {graph_dir}")
    
    # Read protein lists
    with open(os.path.join(splits_dir, 'train_list.txt'), 'r') as f:
        train_proteins = [line.strip() for line in f]
    with open(os.path.join(splits_dir, 'test_list.txt'), 'r') as f:
        test_proteins = [line.strip() for line in f]
    
    all_proteins = train_proteins + test_proteins
    print(f"\nFound {len(all_proteins)} total proteins")
    
    # Collect sizes
    surface_sizes = []
    graph_sizes = []
    protein_ids = []
    
    for protein in tqdm(all_proteins):
        sizes = get_sizes(protein, surface_dir, graph_dir)
        if sizes['surface'] is not None and sizes['graph'] is not None:
            surface_sizes.append(sizes['surface'])
            graph_sizes.append(sizes['graph'])
            protein_ids.append(protein)
    
    surface_sizes = np.array(surface_sizes)
    graph_sizes = np.array(graph_sizes)
    
    # Print statistics
    print("\n=== Surface Statistics ===")
    print(f"Mean: {surface_sizes.mean():.2f}")
    print(f"Median: {np.median(surface_sizes):.2f}")
    print(f"Std dev: {surface_sizes.std():.2f}")
    print(f"Min: {surface_sizes.min()}")
    print(f"Max: {surface_sizes.max()}")
    
    print("\n=== Graph Statistics ===")
    print(f"Mean: {graph_sizes.mean():.2f}")
    print(f"Median: {np.median(graph_sizes):.2f}")
    print(f"Std dev: {graph_sizes.std():.2f}")
    print(f"Min: {graph_sizes.min()}")
    print(f"Max: {graph_sizes.max()}")
    
    # Create visualizations
    plot_size_distribution(surface_sizes, 
                         'Distribution of Surface Sizes (Vertices)',
                         'surface_size_distribution.png')
    plot_size_distribution(graph_sizes,
                         'Distribution of Graph Sizes (Nodes)',
                         'graph_size_distribution.png')
    plot_correlation(surface_sizes, graph_sizes, protein_ids)
    
    # Suggest ranges
    surface_percentiles = np.percentile(surface_sizes, [25, 50, 75])
    graph_percentiles = np.percentile(graph_sizes, [25, 50, 75])
    
    print("\nSuggested filtering ranges:")
    print("\nSurface sizes (vertices):")
    print(f"Small: {surface_sizes.min():.0f} - {surface_percentiles[0]:.0f}")
    print(f"Medium: {surface_percentiles[0]:.0f} - {surface_percentiles[2]:.0f}")
    print(f"Large: {surface_percentiles[2]:.0f} - {surface_sizes.max():.0f}")
    
    print("\nGraph sizes (nodes):")
    print(f"Small: {graph_sizes.min():.0f} - {graph_percentiles[0]:.0f}")
    print(f"Medium: {graph_percentiles[0]:.0f} - {graph_percentiles[2]:.0f}")
    print(f"Large: {graph_percentiles[2]:.0f} - {graph_sizes.max():.0f}")

if __name__ == '__main__':
    data_dir = '/home/tamara/data/masif_site'
    analyze_sizes(data_dir) 