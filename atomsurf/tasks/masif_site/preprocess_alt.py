import os
import sys
import time

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


class PreProcessMSDataset(PreprocessDataset):
    def __init__(self,
                 recompute_s=False,
                 recompute_g=False,
                 data_dir=None,
                 face_reduction_rate=0.5,
                 max_vert_number=100000,
                 use_pymesh=True):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')

        # Set paths
        self.pdb_dir = os.path.join(data_dir, '01-benchmark_pdbs')
        self.ply_dir = os.path.join(data_dir, '01-benchmark_surfaces')
        
        # Get all PDB files
        self.all_pdbs = sorted([os.path.splitext(f)[0] for f in os.listdir(self.pdb_dir) 
                               if f.endswith('.pdb')])
        
        # Create and setup train/test splits
        self._create_data_splits(data_dir)
        
        # Initialize parent class
        super().__init__(data_dir=data_dir, recompute_s=recompute_s, recompute_g=recompute_g,
                         max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate,
                         use_pymesh=use_pymesh)

    def _create_data_splits(self, data_dir):
        """Create initial data splits directory, actual splits will be created later."""
        # Create splits directory
        splits_dir = os.path.join(data_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        # Store paths for later use
        self.train_list_path = os.path.join(splits_dir, 'train_list.txt')
        self.test_list_path = os.path.join(splits_dir, 'test_list.txt')
        
    def create_final_data_splits(self):
        """Create train/test splits using only successfully processed proteins."""
        # Get paths to output directories
        surface_dir = self.out_surf_dir
        rgraph_dir = self.out_rgraph_dir
        
        # Find successfully processed proteins (those with both surface and graph files)
        existing_surfaces = set([os.path.splitext(f)[0] for f in os.listdir(surface_dir) if f.endswith('.pt')])
        existing_graphs = set([os.path.splitext(f)[0] for f in os.listdir(rgraph_dir) if f.endswith('.pt')])
        
        valid_entries = sorted(list(existing_surfaces.intersection(existing_graphs)))
        
        # Generate train/test split (80/20) using only valid entries
        import random
        random.seed(42)  # For reproducibility
        n_total = len(valid_entries)
        n_test = int(0.2 * n_total)
        test_pdbs = sorted(random.sample(valid_entries, n_test))
        train_pdbs = sorted(list(set(valid_entries) - set(test_pdbs)))
        
        print(f"Creating train/test splits with {len(train_pdbs)} training and {len(test_pdbs)} test proteins (80/20 split)")
        
        # Backup original files if they exist
        import shutil
        if os.path.exists(self.train_list_path):
            shutil.copy2(self.train_list_path, self.train_list_path + '.bak')
        if os.path.exists(self.test_list_path):
            shutil.copy2(self.test_list_path, self.test_list_path + '.bak')
        
        # Write new files
        with open(self.train_list_path, 'w') as f:
            f.write('\n'.join(train_pdbs))
        
        with open(self.test_list_path, 'w') as f:
            f.write('\n'.join(test_pdbs))
        
        print("Train/test split files created with successfully processed proteins only")

    def update_data_splits(self):
        """Legacy method, replaced by create_final_data_splits."""
        print("Using create_final_data_splits instead of update_data_splits for better train/test ratio")
        self.create_final_data_splits()

    def get_all_pdbs(self):
        """Override parent class method to use our pre-loaded list"""
        return self.all_pdbs

    def __getitem__(self, idx):
        pdb_name = self.all_pdbs[idx]
        ply_path = os.path.join(self.ply_dir, pdb_name + '.ply')
        surface_dump = os.path.join(self.out_surf_dir, f'{pdb_name}.pt')
        try:
            if self.recompute_s or not os.path.exists(surface_dump):
                # Made a version without pymesh to load the initial data
                if self.use_pymesh:
                    import pymesh
                    mesh = pymesh.load_mesh(ply_path)
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
                # If coarsened, we need to adapt the iface_labels on the new verts
                surface = SurfaceObject.from_verts_faces(verts=verts, faces=faces,
                                                         use_pymesh=self.use_pymesh,
                                                         face_reduction_rate=self.face_reduction_rate)
                if len(surface.verts) < len(iface_labels):
                    old_verts = torch.from_numpy(verts)
                    new_verts = torch.from_numpy(surface.verts)
                    with torch.no_grad():
                        k = 4
                        dists = torch.cdist(old_verts, new_verts)  # (old,new)
                        min_indices = torch.topk(-dists, k=k, dim=0).indices  # (k, new)
                        neigh_labels = torch.sum(iface_labels[min_indices], dim=0) > k / 2
                        # old_fraction = iface_labels.sum() / len(iface_labels)
                        # new_fraction = neigh_labels.sum() / len(neigh_labels)
                        iface_labels = neigh_labels.int()
                surface.iface_labels = iface_labels
                surface.add_geom_feats()
                surface.save_torch(surface_dump)
        except Exception as e:
            print('*******failed******', pdb_name, e)
            return 0
        success = self.name_to_graphs(name=pdb_name)
        return success


if __name__ == '__main__':
    # Process data with different settings
    recompute_g = False
    recompute_s = True
    
    for use_pymesh in (False, True):
        for face_red in [0.1, 0.2, 0.5, 0.9, 1.0]:
            print(f"Processing with use_pymesh={use_pymesh}, face_reduction_rate={face_red}")
            dataset = PreProcessMSDataset(
                recompute_s=recompute_s, 
                recompute_g=recompute_g,
                face_reduction_rate=face_red, 
                use_pymesh=use_pymesh
            )
            do_all(dataset, num_workers=20)
            
            # Create train/test splits with only successfully processed proteins
            dataset.create_final_data_splits()
