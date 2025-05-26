import torch
import os
from pathlib import Path
import pandas as pd
import sys

sys.path.append(os.path.abspath('atomsurf'))
from atomsurf.protein import *  # This imports the necessary classes for loading the data

def print_object_contents(obj, prefix=''):
    """Recursively print the contents of an object"""
    if isinstance(obj, (int, float, str, bool)):
        print(f"{prefix}Value: {obj}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}List/Tuple of length {len(obj)}")
        if len(obj) > 0:
            print(f"{prefix}First element: {type(obj[0])}")
    elif isinstance(obj, dict):
        print(f"{prefix}Dict with keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{prefix}Key '{k}':")
            print_object_contents(v, prefix + '  ')
    elif isinstance(obj, torch.Tensor):
        print(f"{prefix}Tensor shape: {obj.shape}, dtype: {obj.dtype}")
        if obj.numel() > 0:
            print(f"{prefix}First few values: {obj.flatten()[:3]}")
    elif obj is None:
        print(f"{prefix}None")
    else:
        print(f"{prefix}Object of type: {type(obj)}")
        for attr in dir(obj):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    value = getattr(obj, attr)
                    print(f"{prefix}Attribute '{attr}':")
                    print_object_contents(value, prefix + '  ')
                except:
                    print(f"{prefix}Could not access attribute '{attr}'")

def analyze_graph(file_path):
    try:
        data = torch.load(file_path)
        return {
            'file_name': file_path.name,
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'file_size_kb': os.path.getsize(file_path) / 1024,
        }
    except Exception as e:
        print(f"Error loading graph file {file_path}: {e}")
        return None

def analyze_surface(file_path):
    try:
        data = torch.load(file_path)
        return {
            'file_name': file_path.name,
            'num_vertices': len(data.verts) if hasattr(data, 'verts') and data.verts is not None else 0,
            'num_faces': len(data.faces) if hasattr(data, 'faces') and data.faces is not None else 0,
            'file_size_kb': os.path.getsize(file_path) / 1024,
        }
    except Exception as e:
        print(f"Error loading surface file {file_path}: {e}")
        return None

def main():
    # Analyze graphs
    graph_dir = Path('masif_site_data/rgraph')
    surface_dir = Path('masif_site_data/surfaces_0.1_False')
    
    print("\n=== Analyzing all graph files ===")
    graph_stats = []
    for file in graph_dir.glob('*.pt'):
        stats = analyze_graph(file)
        if stats:
            graph_stats.append(stats)
    
    print("\n=== Analyzing all surface files ===")
    surface_stats = []
    for file in surface_dir.glob('*.pt'):
        stats = analyze_surface(file)
        if stats:
            surface_stats.append(stats)
    
    if graph_stats:
        # Convert to pandas DataFrame and sort by number of nodes
        graph_df = pd.DataFrame(graph_stats)
        graph_df = graph_df.sort_values('num_nodes')
        
        print("\n=== Graph Statistics ===")
        print("\nSmallest 10 graphs (by node count):")
        print(graph_df[['file_name', 'num_nodes', 'num_edges']].head(1000))
        
        print("\nGraph size distribution:")
        print(graph_df['num_nodes'].describe())
    
    if surface_stats:
        # Convert to pandas DataFrame and sort by number of vertices
        surface_df = pd.DataFrame(surface_stats)
        surface_df = surface_df.sort_values('num_vertices')
        
        print("\n=== Surface Statistics ===")
        print("\nSmallest 10 surfaces (by vertex count):")
        print(surface_df[['file_name', 'num_vertices', 'num_faces']].head(10))
        
        print("\nSurface size distribution:")
        print(surface_df['num_vertices'].describe())
        
if __name__ == "__main__":
    main() 