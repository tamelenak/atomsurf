"""
MaSIF Site Task Tutorial
=======================

This script demonstrates how to use AtomSurf for the MaSIF site task,
which involves predicting protein-protein interaction sites on protein surfaces.

The workflow includes:
1. Data preprocessing
2. Model training
3. Testing and visualization
"""

import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig
import torch.nn as nn
import io
import warnings

# Import the Features class
from atomsurf.protein.features import Features
from atomsurf.protein.create_esm import get_esm_embedding_single, get_esm_embedding_batch
from atomsurf.utils.data_utils import AtomBatch, PreprocessDataset, SurfaceLoader, GraphLoader
from atomsurf.utils.python_utils import do_all
from atomsurf.utils.wrappers import DefaultLoader, get_default_model
from atomsurf.tasks.masif_site.preprocess import PreProcessMSDataset
from atomsurf.tasks.masif_site.model import MasifSiteNet
from atomsurf.tasks.masif_site.data_loader import MasifSiteDataset

# Monkey patch the Features class to silence the specific warnings
original_build_expanded_features = Features.build_expanded_features

def silent_build_expanded_features(self, feature_keys='all', oh_keys='all', feature_expander=None):
    # Temporarily redirect stdout
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # Call the original method
    result = original_build_expanded_features(self, feature_keys, oh_keys, feature_expander)
    
    # Restore stdout
    sys.stdout = original_stdout
    
    return result

# Apply the patch
Features.build_expanded_features = silent_build_expanded_features

# Disable print output for warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Redirect stdout context manager - no longer needed with our patch
class SuppressOutput:
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
        
    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        return self
        
    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def setup_directories():
    """Set up necessary directories for data processing and results."""
    data_dir = "data/masif_site"
    benchmark_pdb_dir = os.path.join(data_dir, "01-benchmark_pdbs")
    # Surface directory will include the face reduction rate in its name
    surface_dir = os.path.join(data_dir, "surfaces_0.5_False")  # 0.5 is the face_reduction_rate, False for use_pymesh
    rgraph_dir = os.path.join(data_dir, "rgraph")
    esm_dir = os.path.join(data_dir, "esm_emb")

    # Create output directories if they don't exist
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(rgraph_dir, exist_ok=True)
    os.makedirs(esm_dir, exist_ok=True)

    return data_dir, benchmark_pdb_dir, surface_dir, rgraph_dir, esm_dir


def preprocess_data(data_dir, pdb_dir, esm_dir):
    """Preprocess the data including surface generation and ESM embeddings."""
    print("Starting data preprocessing...")
    
    # Initialize the preprocessing dataset
    dataset = PreProcessMSDataset(
        data_dir=data_dir,
        recompute_s=True,  # Set to True to recompute surfaces
        recompute_g=True,  # Set to True to recompute graphs
        face_reduction_rate=0.5,  # Adjust this value to control mesh resolution
        use_pymesh=True
    )

    # Run preprocessing
    print("Processing surfaces and graphs...")
    do_all(dataset, num_workers=4)  # Adjust number of workers based on your system

    # Generate ESM embeddings
    print("Generating ESM embeddings...")
    get_esm_embedding_batch(in_pdbs_dir=pdb_dir, dump_dir=esm_dir)
    
    print("Preprocessing complete!")


def setup_datasets(data_dir, surface_dir, rgraph_dir, esm_dir):
    """Set up training and testing datasets."""
    # Create configuration objects for surface and graph loaders
    
    # Surface configuration
    cfg_surface = DictConfig({})
    cfg_surface.use_surfaces = True
    cfg_surface.feat_keys = 'all'
    cfg_surface.oh_keys = 'all'
    cfg_surface.gdf_expand = True
    cfg_surface.data_dir = data_dir  # The root data directory
    cfg_surface.data_name = 'surfaces_0.5_False'  # The subdirectory containing the .pt files
    
    # Graph configuration
    cfg_graph = DictConfig({})
    cfg_graph.use_graphs = True
    cfg_graph.feat_keys = 'all'
    cfg_graph.oh_keys = 'all'
    cfg_graph.esm_dir = esm_dir
    # Re-enable ESM features now that we've fixed the one-hot encoding issue
    cfg_graph.use_esm = True  # Changed back to True
    cfg_graph.data_dir = data_dir  # The root data directory
    cfg_graph.data_name = 'rgraph'  # The subdirectory containing the graph .pt files
    
    # Create surface and graph builders
    surface_builder = SurfaceLoader(cfg_surface)
    graph_builder = GraphLoader(cfg_graph)
    
    # Load the training systems list
    train_systems_list = os.path.join(data_dir, 'splits', 'train_list.txt')
    train_systems = [name.strip() for name in open(train_systems_list, 'r').readlines()]
    
    # Filter train systems to keep only the ones where files exist
    print("Filtering systems to remove missing files...")
    valid_train_systems = []
    for system in train_systems:
        graph_path = os.path.join(rgraph_dir, f"{system}.pt")
        surface_path = os.path.join(surface_dir, f"{system}.pt")
        if os.path.exists(graph_path) and os.path.exists(surface_path):
            valid_train_systems.append(system)
    
    print(f"Found {len(valid_train_systems)} valid training systems out of {len(train_systems)}")
    
    # Load the test systems list
    test_systems_list = os.path.join(data_dir, 'splits', 'test_list.txt')
    test_systems = [name.strip() for name in open(test_systems_list, 'r').readlines()]
    
    # Filter test systems to keep only the ones where files exist
    valid_test_systems = []
    for system in test_systems:
        graph_path = os.path.join(rgraph_dir, f"{system}.pt")
        surface_path = os.path.join(surface_dir, f"{system}.pt")
        if os.path.exists(graph_path) and os.path.exists(surface_path):
            valid_test_systems.append(system)
    
    print(f"Found {len(valid_test_systems)} valid test systems out of {len(test_systems)}")
    
    if not valid_train_systems:
        raise ValueError("No valid training systems found with existing files!")
    
    # Create datasets with filtered systems
    train_dataset = MasifSiteDataset(
        systems=valid_train_systems,
        surface_builder=surface_builder,
        graph_builder=graph_builder
    )
    
    # Find a valid example quietly
    for i in range(min(5, len(valid_train_systems))):
        system = valid_train_systems[i]
        surface = surface_builder.load(system)
        graph = graph_builder.load(system)
        if surface is not None and graph is not None:
            train_dataset.valid_idx = i
            break
    else:
        raise ValueError("No valid data found in the dataset")
    
    test_dataset = MasifSiteDataset(
        systems=valid_test_systems,
        surface_builder=surface_builder,
        graph_builder=graph_builder
    )
    
    # Create training data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=AtomBatch.from_data_list
    )
    
    return train_dataset, test_dataset, train_loader


def setup_model(train_dataset):
    """Initialize and set up the MaSIF site model."""
    # Get input dimensions from example data
    if hasattr(train_dataset, 'valid_idx'):
        example_data = train_dataset[train_dataset.valid_idx]
    else:
        # Try to find a valid example
        for i in range(len(train_dataset)):
            example_data = train_dataset[i]
            if example_data is not None:
                break
        else:
            raise ValueError("No valid data found in the dataset")
    
    # Create proper configuration for encoder
    cfg_encoder = DictConfig({
        "blocks": []  # Empty blocks list - the model will handle this internally
    })
    
    # Create configuration for head
    cfg_head = DictConfig({
        "encoded_dims": 52,  # Changed from 32 to 52 to match the actual feature dimensions
        "output_dims": 1
    })

    # Initialize the original MasifSiteNet model
    model = MasifSiteNet(
        cfg_encoder=cfg_encoder,
        cfg_head=cfg_head
    )

    # Set up device and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    return model, device, optimizer, criterion


def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10):
    """Train the MaSIF site model."""
    print("Starting model training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        processed_items = 0
        error_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                pred = model(batch)
                # Flatten the predictions to match the target shape
                pred = pred.x.flatten()
                loss = criterion(pred, batch.surface.iface_labels.float())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_size = len(pred) if hasattr(pred, '__len__') else 1
                processed_items += batch_size
                total_loss += loss.item()
                batch_count += 1
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f'  Batch {batch_idx}/{len(train_loader)}, Current loss: {loss.item():.4f}')
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing batch {batch_idx}: {e}")
                if error_count > 10:
                    print("Too many errors, stopping training")
                    break
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Processed items: {processed_items}, Error batches: {error_count}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, No valid batches processed')
    
    print("Training complete!")


def visualize_predictions(vertices, faces, predictions, true_labels=None):
    """Visualize the predicted and true interaction sites on the protein surface."""
    fig = plt.figure(figsize=(12, 6))
    
    # Plot predictions
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                     triangles=faces, cmap='coolwarm',
                     array=predictions.cpu().numpy())
    ax1.set_title('Predictions')
    
    if true_labels is not None:
        # Plot true labels
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                         triangles=faces, cmap='coolwarm',
                         array=true_labels.cpu().numpy())
        ax2.set_title('True Labels')
    
    plt.show()


def evaluate_model(model, test_dataset, device):
    """Evaluate the model on test data and visualize results."""
    print("Evaluating model...")
    
    model.eval()
    with torch.no_grad():
        # Find a valid test example
        for i in range(min(5, len(test_dataset))):
            test_data = test_dataset[i]
            
            if test_data is not None:
                print(f"Using test example {i} for evaluation")
                test_data = test_data.to(device)
                batch = AtomBatch.from_data_list([test_data])
                pred = model(batch)
                pred_labels = (torch.sigmoid(pred.x) > 0.5).float()
                
                # Visualize results
                visualize_predictions(
                    vertices=test_data.surface.pos.cpu(),
                    faces=test_data.surface.face.t().cpu(),
                    predictions=pred_labels,
                    true_labels=test_data.surface.iface_labels
                )
                break
        else:
            print("Could not find a valid test example for evaluation")


def main():
    """Main function to run the complete MaSIF site workflow."""
    # Setup
    data_dir, pdb_dir, surface_dir, rgraph_dir, esm_dir = setup_directories()
    
    # Preprocessing
    #preprocess_data(data_dir, pdb_dir, esm_dir)
    
    # Dataset setup - our monkey patch will suppress the specific warnings
    print("Loading datasets...")
    train_dataset, test_dataset, train_loader = setup_datasets(
        data_dir, surface_dir, rgraph_dir, esm_dir
    )
    
    # Model setup
    print("Setting up model...")
    model, device, optimizer, criterion = setup_model(train_dataset)
    
    # Training
    train_model(model, train_loader, optimizer, criterion, device)
    
    # Evaluation and visualization
    evaluate_model(model, test_dataset, device)


if __name__ == "__main__":
    main() 