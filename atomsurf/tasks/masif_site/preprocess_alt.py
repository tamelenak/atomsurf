import os
import sys
import numpy as np
import torch

# For script execution
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


class PreProcessMSDataset(PreprocessDataset):
    """Dataset for preprocessing MaSIF site data including surface mesh generation and interface labels."""
    
    def __init__(self,
                 recompute_s=False,
                 recompute_g=False,
                 data_dir=None,
                 face_reduction_rate=0.5,
                 max_vert_number=100000,
                 use_pymesh=True):
        """Initialize the MaSIF site preprocessing dataset.
        
        Args:
            recompute_s: Whether to recompute surfaces
            recompute_g: Whether to recompute graphs
            data_dir: Root data directory (default: ../../../data/masif_site)
            face_reduction_rate: Rate for mesh simplification (0-1)
            max_vert_number: Maximum number of vertices
            use_pymesh: Whether to use pymesh for processing
        """
        # Set default data directory if not provided
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
        """Create train/test splits if they don't exist."""
        # Create splits directory
        splits_dir = os.path.join(data_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        # Paths for train/test lists
        train_list = os.path.join(splits_dir, 'train_list.txt')
        test_list = os.path.join(splits_dir, 'test_list.txt')
        
        # Create splits if they don't exist
        if not os.path.exists(train_list) or not os.path.exists(test_list):
            import random
            random.seed(42)  # For reproducibility
            
            # Generate train/test split (80/20)
            n_total = len(self.all_pdbs)
            n_test = int(0.2 * n_total)
            test_pdbs = sorted(random.sample(self.all_pdbs, n_test))
            train_pdbs = sorted(list(set(self.all_pdbs) - set(test_pdbs)))
            
            # Save splits
            with open(train_list, 'w') as f:
                f.write('\n'.join(train_pdbs))
            with open(test_list, 'w') as f:
                f.write('\n'.join(test_pdbs))

    def get_all_pdbs(self):
        """Override parent class method to use our pre-loaded list."""
        return self.all_pdbs

    def __getitem__(self, idx):
        """Process a single protein at index idx."""
        pdb_name = self.all_pdbs[idx]
        ply_path = os.path.join(self.ply_dir, f"{pdb_name}.ply")
        surface_dump = os.path.join(self.out_surf_dir, f"{pdb_name}.pt")
        
        try:
            if self.recompute_s or not os.path.exists(surface_dump):
                # Load mesh data based on availability of pymesh
                mesh_data = self._load_mesh_data(ply_path)
                if mesh_data is None:
                    return 0
                    
                verts, faces, iface_labels = mesh_data
                iface_labels = torch.from_numpy(iface_labels)
                
                # Create surface from verts and faces
                surface = SurfaceObject.from_verts_faces(
                    verts=verts, faces=faces,
                    use_pymesh=self.use_pymesh,
                    face_reduction_rate=self.face_reduction_rate
                )
                
                # Handle interface labels for reduced mesh
                if len(surface.verts) < len(iface_labels):
                    iface_labels = self._transfer_labels(
                        verts, surface.verts, iface_labels
                    )
                
                # Add labels and features to surface
                surface.iface_labels = iface_labels
                surface.add_geom_feats()
                surface.save_torch(surface_dump)
                
        except Exception as e:
            print(f"Failed to process {pdb_name}: {e}")
            return 0
            
        # Process graphs
        success = self.name_to_graphs(name=pdb_name)
        return success
        
    def _load_mesh_data(self, ply_path):
        """Load mesh data from a PLY file using pymesh or plyfile."""
        try:
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
            return verts, faces, iface_labels
        except Exception as e:
            print(f"Error loading mesh from {ply_path}: {e}")
            return None
            
    def _transfer_labels(self, old_verts, new_verts, old_labels, k=4):
        """Transfer labels from original mesh to reduced mesh using nearest neighbors."""
        with torch.no_grad():
            old_verts = torch.from_numpy(old_verts)
            new_verts = torch.from_numpy(new_verts)
            
            # Find k nearest neighbors for each new vertex
            dists = torch.cdist(old_verts, new_verts)  # (old,new)
            min_indices = torch.topk(-dists, k=k, dim=0).indices  # (k, new)
            
            # Majority vote for labels
            neigh_labels = torch.sum(old_labels[min_indices], dim=0) > k / 2
            return neigh_labels.int()


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
