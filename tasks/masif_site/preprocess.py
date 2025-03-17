import os
import sys
import time
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
import trimesh

# Set multiprocessing start method to 'spawn'
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.utils.data_utils import PreprocessDataset, pdb_to_surf, pdb_to_graphs
from atomsurf.utils.python_utils import do_all

# Set up CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)

def process_single_cpu(pdb_name, dataset):
    """Process a single PDB file using CPU only"""
    ply_path = os.path.join(dataset.ply_dir, pdb_name + '.ply')
    surface_dump = os.path.join(dataset.out_surf_dir, f'{pdb_name}.pt')
    try:
        if dataset.recompute_s or not os.path.exists(surface_dump):
            # Load mesh data
            if dataset.use_pymesh:
                mesh = trimesh.load(ply_path)
                verts = mesh.vertices.astype(np.float32)
                faces = mesh.faces.astype(np.int32)
                iface_labels = mesh.get_attribute("vertex_iface").astype(np.int32)
            else:
                from plyfile import PlyData
                with open(ply_path, 'rb') as f:
                    plydata = PlyData.read(f)
                    vx = plydata['vertex']['x']
                    vy = plydata['vertex']['y']
                    vz = plydata['vertex']['z']
                    verts = np.stack((vx, vy, vz), axis=1).astype(np.float32)
                    faces = np.stack(plydata['face']['vertex_indices'], axis=0).astype(np.int32)
                    iface_labels = plydata['vertex']['iface'].astype(np.int32)
            iface_labels = torch.from_numpy(iface_labels)
            
            # Create surface object (CPU only)
            surface = SurfaceObject.from_verts_faces(verts=verts, faces=faces,
                                                   use_pymesh=dataset.use_pymesh,
                                                   face_reduction_rate=dataset.face_reduction_rate)
            surface.add_geom_feats()
            surface.save_torch(surface_dump)
            return 1
        return 1
    except Exception as e:
        print(f"\nFailed to process {pdb_name}: {str(e)}", file=sys.stderr)  # Print errors to stderr
        return 0

class PreProcessMSDataset(PreprocessDataset):
    def __init__(self,
                 recompute_s=False,
                 recompute_g=False,
                 data_dir=None,
                 face_reduction_rate=0.5,
                 max_vert_number=100000,
                 use_pymesh=True,
                 test_split=0.2,
                 random_seed=42):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')

        # Initialize without calling parent class __init__
        self.pdb_dir = os.path.join(data_dir, '01-benchmark_pdbs')
        self.ply_dir = os.path.join(data_dir, '01-benchmark_surfaces')

        # Set up surface parameters
        self.max_vert_number = max_vert_number
        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh
        self.recompute_s = recompute_s
        self.recompute_g = recompute_g

        # Set up output directories
        surface_dirname = f'surfaces_{face_reduction_rate}{f"_{use_pymesh}" if use_pymesh is not None else ""}'
        self.out_surf_dir = os.path.join(data_dir, surface_dirname)
        os.makedirs(self.out_surf_dir, exist_ok=True)

        self.out_rgraph_dir = os.path.join(data_dir, 'rgraph')
        os.makedirs(self.out_rgraph_dir, exist_ok=True)

        # Get all PDB files and create train/test splits
        all_pdb_files = [f[:-4] for f in os.listdir(self.pdb_dir) if f.endswith('.pdb')]
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Shuffle and split
        random.shuffle(all_pdb_files)
        test_size = int(len(all_pdb_files) * test_split)
        test_names = set(all_pdb_files[:test_size])
        train_names = set(all_pdb_files[test_size:])
        
        # Save splits for future use
        splits_dir = os.path.join(data_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        with open(os.path.join(splits_dir, 'train_list.txt'), 'w') as f:
            f.write('\n'.join(sorted(train_names)))
        with open(os.path.join(splits_dir, 'test_list.txt'), 'w') as f:
            f.write('\n'.join(sorted(test_names)))
            
        self.all_pdbs = sorted(all_pdb_files)

    def process_all(self, num_workers=4, use_gpu=False):
        """Process all PDB files with proper multiprocessing"""
        process_func = process_single_gpu if use_gpu else process_single_cpu
        
        # Use ProcessPoolExecutor with 'spawn' start method
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Create partial function with dataset
            process_func_with_dataset = partial(process_func, dataset=self)
            
            # Process all files with progress tracking
            total_files = len(self.all_pdbs)
            print(f"\nStarting processing of {total_files} files...")
            results = []
            total_success = 0
            t0 = time.time()
            
            # Use executor.map to process files
            for i, success in enumerate(executor.map(process_func_with_dataset, self.all_pdbs)):
                results.append(success)
                total_success += int(success)
                # Print progress frequently with carriage return
                print(f'\rProcessing: {i + 1}/{total_files} ({(i + 1)/total_files*100:.1f}%) | Success: {total_success} | Time: {time.time() - t0:.1f}s', end='', flush=True)
            
            # Print final summary on a new line
            print(f'\n\nCompleted processing {total_files} files with {total_success} successes in {time.time() - t0:.1f}s') 