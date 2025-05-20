# std
import sys
from pathlib import Path
import os
# 3p
import hydra
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from atomsurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger
from pl_model import MasifSiteModule
from data_loader import MasifSiteDataModule
import warnings 
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

from atomsurf.tasks.masif_site.data_loader import MasifSiteDataset
from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader

# --- CONFIGURATION ---
# Set these to match your setup
data_dir = "/path/to/your/masif_site_data"  # Replace with your actual data directory
surface_dir = os.path.join(data_dir, "surfaces_0.1_False")
graph_dir = os.path.join(data_dir, "rgraph")
split_file = os.path.join(data_dir, "splits/train_list.txt")

# Optionally, you can uncomment the next line and specify an actual path directly
# data_dir = "/root/atomsurf/data/masif_site"

# Load protein IDs from split file (if file exists)
protein_ids = []
if os.path.exists(split_file):
    with open(split_file, "r") as f:
        protein_ids = [line.strip() for line in f]
else:
    # If no split file, test with a single protein
    protein_ids = ['5G1X_A']

# Create loaders with proper configuration
class DummyConfig:
    def __init__(self, base_dir, data_name=""):
        self.data_dir = os.path.dirname(base_dir)  # Parent directory
        self.data_name = os.path.basename(base_dir) # Directory name
        self.use_surfaces = True
        self.feat_keys = 'all'
        self.oh_keys = 'all'
        self.gdf_expand = True
        self.use_graphs = True
        self.use_esm = False
        self.esm_dir = os.path.join(data_dir, '01-benchmark_esm_embs')

# Initialize loaders with proper directory structure
surface_loader = SurfaceLoader(DummyConfig(surface_dir))
graph_loader = GraphLoader(DummyConfig(graph_dir))

# Print path information for debugging
print(f"Surface loader data_dir: {surface_loader.data_dir}")
print(f"Graph loader data_dir: {graph_loader.data_dir}")

# Create dataset
dataset = MasifSiteDataset(protein_ids, surface_loader, graph_loader)

# Try loading a few items
for i in range(min(len(protein_ids), 10)):
    print(f"\n--- Loading index {i} ({protein_ids[i]}) ---")
    try:
        item = dataset[i]
        if item is not None:
            print(f"Result: Item loaded successfully with surface verts: {item.surface.verts.shape if hasattr(item.surface, 'verts') else 'None'}")
        else:
            print("Result: None - failed to load item")
    except Exception as e:
        print(f"Error loading item: {e}")

@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init datamodule
    datamodule = MasifSiteDataModule(cfg)
    
    # Test loading a few proteins
    train_sys = datamodule.train_sys[:10]  # Just test first 10
    print(f"\nTesting protein loading for {len(train_sys)} proteins")
    
    for i, pocket in enumerate(train_sys):
        print(f"\n--- Loading protein {i}: {pocket} ---")
        surface = datamodule.surface_loader.load(pocket)
        graph = datamodule.graph_loader.load(pocket)
        print(f"Surface loaded: {surface is not None}, Graph loaded: {graph is not None}")
        
        if surface is not None and hasattr(surface, 'verts'):
            print(f"Surface vertices: {surface.verts.shape}")
            print(f"Surface has features: {hasattr(surface, 'x')}")
        
    print("\nLoading test complete")

if __name__ == "__main__":
    main() 