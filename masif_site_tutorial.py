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

from atomsurf.protein.create_esm import get_esm_embedding_single, get_esm_embedding_batch
from atomsurf.utils.data_utils import AtomBatch, PreprocessDataset
from atomsurf.utils.python_utils import do_all
from atomsurf.utils.wrappers import DefaultLoader, get_default_model
from atomsurf.tasks.masif_site.preprocess import PreProcessMSDataset
from atomsurf.tasks.masif_site.model import MasifSiteModel
from atomsurf.tasks.masif_site.data_loader import MasifSiteDataset


def setup_directories():
    """Set up necessary directories for data processing and results."""
    data_dir = "data/masif_site"
    benchmark_pdb_dir = os.path.join(data_dir, "01-benchmark_pdbs")
    surface_dir = os.path.join(data_dir, "surfaces")
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
    # Load the training dataset
    train_dataset = MasifSiteDataset(
        data_dir=data_dir,
        split='train',
        surface_dir=surface_dir,
        graph_dir=rgraph_dir,
        embeddings_dir=esm_dir
    )

    # Create training data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=AtomBatch.from_data_list
    )

    # Load the test dataset
    test_dataset = MasifSiteDataset(
        data_dir=data_dir,
        split='test',
        surface_dir=surface_dir,
        graph_dir=rgraph_dir,
        embeddings_dir=esm_dir
    )

    return train_dataset, train_loader, test_dataset


def setup_model(train_dataset):
    """Initialize and set up the MaSIF site model."""
    # Get input dimensions from example data
    example_data = train_dataset[0]
    in_dim_surface = example_data.surface.x.shape[-1]
    in_dim_graph = example_data.graph.x.shape[-1]

    # Initialize model
    model = MasifSiteModel(
        in_dim_surface=in_dim_surface,
        in_dim_graph=in_dim_graph,
        hidden_dim=32
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
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch)
            loss = criterion(pred, batch.surface.iface_labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
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
        test_data = test_dataset[0].to(device)
        pred = model(AtomBatch.from_data_list([test_data]))
        pred_labels = (torch.sigmoid(pred) > 0.5).float()

    # Visualize results
    visualize_predictions(
        vertices=test_data.surface.pos.cpu(),
        faces=test_data.surface.face.t().cpu(),
        predictions=pred_labels,
        true_labels=test_data.surface.iface_labels
    )


def main():
    """Main function to run the complete MaSIF site workflow."""
    # Setup
    data_dir, pdb_dir, surface_dir, rgraph_dir, esm_dir = setup_directories()
    
    # Preprocessing
    preprocess_data(data_dir, pdb_dir, esm_dir)
    
    # Dataset setup
    train_dataset, train_loader, test_dataset = setup_datasets(
        data_dir, surface_dir, rgraph_dir, esm_dir
    )
    
    # Model setup
    model, device, optimizer, criterion = setup_model(train_dataset)
    
    # Training
    train_model(model, train_loader, optimizer, criterion, device)
    
    # Evaluation and visualization
    evaluate_model(model, test_dataset, device)


if __name__ == "__main__":
    main() 