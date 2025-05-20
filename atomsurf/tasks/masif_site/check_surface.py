import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add the parent directory to the path so 'atomsurf' can be imported
sys.path.append(str(Path(__file__).absolute().parents[3]))

from atomsurf.utils.data_utils import SurfaceLoader

def check_surface_file(protein_id, cfg_surface):
    surface_path = os.path.join(cfg_surface.data_dir, cfg_surface.data_name, f"{protein_id}.pt")
    print(f"Checking surface file: {surface_path}")
    if os.path.exists(surface_path):
        print(f"  ✓ Surface file exists")
        try:
            raw_data = torch.load(surface_path)
            print(f"  ✓ Successfully loaded raw data")
            print(f"  Data type: {type(raw_data)}")
            if hasattr(raw_data, 'verts'):
                print(f"  ✓ Has vertices: {raw_data.verts.shape if hasattr(raw_data.verts, 'shape') else 'No shape'}")
            else:
                print(f"  ✗ Missing vertices attribute")
            loader = SurfaceLoader(cfg_surface)
            surface = loader.load(protein_id)
            if surface is not None:
                print(f"  ✓ Successfully loaded with SurfaceLoader")
                print(f"  Surface contains {len(surface.verts)} vertices")
                return True
            else:
                print(f"  ✗ SurfaceLoader returned None")
                return False
        except Exception as e:
            print(f"  ✗ Error loading surface: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"  ✗ Surface file does not exist")
        return False

def main():
    protein_ids = ["3GPG_D", "1GRC_B", "5G1X_A", "2BOL_B", "1GRC_A"]
    # Use OmegaConf for config, just like in your training script
    cfg_surface = OmegaConf.create({
        "use_surfaces": True,
        "data_dir": "/root/atomsurf/masif_site_data",
        "data_name": "surfaces_0.1_False",
        "feat_keys": "all",
        "oh_keys": "all",
        "gdf_expand": True
    })
    print(f"Surface data directory: {os.path.join(cfg_surface.data_dir, cfg_surface.data_name)}")
    for protein_id in protein_ids:
        print(f"\nChecking protein {protein_id}:")
        check_surface_file(protein_id, cfg_surface)
    surface_dir = os.path.join(cfg_surface.data_dir, cfg_surface.data_name)
    if os.path.exists(surface_dir):
        print(f"\nListing some files in {surface_dir}:")
        files = os.listdir(surface_dir)[:10]
        for file in files:
            print(f"  {file}")

if __name__ == "__main__":
    main()