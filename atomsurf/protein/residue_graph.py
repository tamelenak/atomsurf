import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path, atom_coords_to_edges, res_type_to_hphob
from atomsurf.protein.features import Features, FeaturesHolder

from atomsurf.protein.create_esm import get_esm_embedding_single
from atomsurf.utils.torch_utils import safe_to_torch


class PronetFeaturesComputer:
    """
    adapted from https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/dataset/ECdataset.py
    """

    def __init__(self):
        pass

    @staticmethod
    def _normalize(tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    @staticmethod
    def get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, 'N')
        mask_ca = np.char.equal(atom_names, 'CA')
        mask_c = np.char.equal(atom_names, 'C')
        mask_cb = np.char.equal(atom_names, 'CB')  # This was wrong
        mask_g = np.char.equal(atom_names, 'CG') | np.char.equal(atom_names, 'SG') | np.char.equal(atom_names,
                                                                                                   'OG') | np.char.equal \
                     (atom_names, 'CG1') | np.char.equal(atom_names, 'OG1')
        mask_d = np.char.equal(atom_names, 'CD') | np.char.equal(atom_names, 'SD') | np.char.equal(atom_names,
                                                                                                   'CD1') | np.char.equal \
                     (atom_names, 'OD1') | np.char.equal(atom_names, 'ND1')
        mask_e = np.char.equal(atom_names, 'CE') | np.char.equal(atom_names, 'NE') | np.char.equal(atom_names, 'OE1')
        mask_z = np.char.equal(atom_names, 'CZ') | np.char.equal(atom_names, 'NZ')
        mask_h = np.char.equal(atom_names, 'NH1')

        pos_n = np.full((len(amino_types), 3), np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types), 3), np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types), 3), np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types), 3), np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types), 3), np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types), 3), np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types), 3), np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types), 3), np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types), 3), np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h

    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_dihedrals(v1, v2, v3), 1)
        angle2 = torch.unsqueeze(self.compute_dihedrals(v2, v3, v4), 1)
        angle3 = torch.unsqueeze(self.compute_dihedrals(v3, v4, v5), 1)
        angle4 = torch.unsqueeze(self.compute_dihedrals(v4, v5, v6), 1)
        angle5 = torch.unsqueeze(self.compute_dihedrals(v5, v6, v7), 1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4), 1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)), 1)

        return side_chain_embs

    def bb_embs(self, X):
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_dihedrals(u0, u1, u2)

        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2])
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    def compute_dihedrals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion

    def get_pronet_features(self, amino_types, atom_amino_id, atom_names, atom_pos):
        amino_types = np.asarray(amino_types)
        atom_amino_id = np.asarray(atom_amino_id)
        atom_names = np.asarray(atom_names)
        atom_pos = np.asarray(atom_pos)
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(amino_types, atom_names,
                                                                                            atom_amino_id, atom_pos)

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(
            torch.cat((torch.unsqueeze(pos_n, 1), torch.unsqueeze(pos_ca, 1), torch.unsqueeze(pos_c, 1)), 1))
        bb_embs[torch.isnan(bb_embs)] = 0

        # Save those results now
        data = Data()
        data.side_chain_embs = side_chain_embs
        data.bb_embs = bb_embs
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c
        return data


class ResidueGraph(Data, FeaturesHolder):
    def __init__(self, node_pos, edge_index=None, edge_attr=None, features=None, node_names=None, **kwargs):
        edge_index = safe_to_torch(edge_index)
        edge_attr = safe_to_torch(edge_attr)
        super(ResidueGraph, self).__init__(edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        self.node_pos = safe_to_torch(node_pos)
        self.num_res = len(node_pos) if node_pos is not None else 0
        self.node_names = node_names
        # Useful for bipartite computations
        self.node_len = self.num_res
        if features is None:
            self.features = Features(num_nodes=self.num_res)
        else:
            self.features = features


class RGraphBatch(Batch):
    """
    This class is useful for PyG Batching

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def batch_from_data_list(cls, data_list):
        batch = Batch.from_data_list(data_list)
        batch = batch.contiguous()
        surface_batch = cls()
        surface_batch.__dict__.update(batch.__dict__)
        return surface_batch


class ResidueGraphBuilder:
    def __init__(self, add_pronet=True, add_esm=False):
        self.add_pronet = add_pronet
        self.add_esm = add_esm
        pass

    def arrays_to_resgraph(self, arrays, pdb_path=None, esm_path=None):
        """
        Creation from array directly enables a speed up that originates from loading pdb once for both agraph anr rgraph
        :param arrays:
        :param pdb_path:
        :param esm_path:
        :return:
        """
        amino_types, atom_chain_id, atom_amino_id, atom_names, _, atom_pos, atom_charge, atom_radius, res_sse, amino_ids, _ = arrays
        mask_ca = np.array([name == 'CA' for name in atom_names])
        pos_ca = np.full((len(amino_types), 3), np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        edge_index, edge_dists = atom_coords_to_edges(pos_ca, edge_dist_cutoff=12)

        res_graph = ResidueGraph(node_pos=pos_ca,
                                 edge_index=edge_index,
                                 edge_attr=edge_dists,
                                 node_names=amino_ids)
        res_graph.features.add_named_oh_features('amino_types', amino_types, 21)
        res_graph.features.add_named_oh_features('sse', res_sse, nclasses=8)
        hphob = np.asarray([res_type_to_hphob[amino_type] for amino_type in amino_types], dtype=np.float32)
        res_graph.features.add_named_features('hphobs', hphob)
        if self.add_esm:
            esm_embed = get_esm_embedding_single(pdb_path, esm_path)
            res_graph.features.add_named_features('esm_embed', esm_embed)
        if self.add_pronet:
            pfc = PronetFeaturesComputer()
            pronet_features = pfc.get_pronet_features(amino_types, atom_amino_id, atom_names, atom_pos)
            res_graph.features.add_misc_features("pronet_features", pronet_features)
        return res_graph
    
    def arrays_to_resgraph2(self, arrays, pdb_path=None, esm_path=None):
        """
        Creation from array directly enables a speed up that originates from loading pdb once for both agraph anr rgraph
        :param arrays:
        :param pdb_path:
        :param esm_path:
        :return:
        """
        # Print the length of arrays tuple to debug
        print(f"Arrays tuple length: {len(arrays)}")
        
        # Correct unpacking
        if len(arrays) >= 10:
            amino_types = arrays[0]
            atom_chain_id = arrays[1]
            atom_amino_id = arrays[2]
            atom_names = arrays[3]
            atom_pos = arrays[5]
            atom_charge = arrays[6]
            atom_radius = arrays[7]
            res_sse = arrays[8]
            amino_ids = arrays[9]
        else:
            raise ValueError(f"Expected at least 10 elements in arrays tuple, got {len(arrays)}")
        
        # Check the type of atom_names and handle accordingly
        print(f"atom_names type: {type(atom_names)}, dtype: {atom_names.dtype}")
        
        # If atom_names are integers, we need a mapping to identify CA atoms
        # Common mapping: 0="N", 1="CA", 2="C", 3="O", etc.
        # This is a guess - you might need to adjust based on your actual data
        if np.issubdtype(atom_names.dtype, np.integer) or np.issubdtype(atom_names.dtype, np.int32):
            print("Atom names are integers, attempting to identify CA atoms by convention")
            # Assuming 1 might be CA (this is a guess - adjust as needed)
            mask_ca = atom_names == 1
        elif np.issubdtype(atom_names.dtype, np.str_):
            # Normal string comparison
            atom_names = atom_names.tolist()
            mask_ca = np.array([name == 'CA' for name in atom_names])
        else:
            print(f"Unexpected atom_names dtype: {atom_names.dtype}")
            # Try a few common mappings for CA atoms
            possible_ca_values = [1, 2, 3, 4]  # Try different common values
            for ca_value in possible_ca_values:
                mask_ca = atom_names == ca_value
                if np.sum(mask_ca) > 0:
                    print(f"Found potential CA atoms with value {ca_value}: {np.sum(mask_ca)} atoms")
                    break
            else:
                # If no CA atoms found with any value, create empty mask
                print("Could not identify any CA atoms!")
                mask_ca = np.zeros(len(atom_names), dtype=bool)
        
        print(f"CA atoms found: {np.sum(mask_ca)}")
        
        # Debug atom_amino_id
        print(f"atom_amino_id shape: {atom_amino_id.shape}")
        print(f"atom_amino_id min: {atom_amino_id.min()}, max: {atom_amino_id.max()}")
        print(f"amino_types length: {len(amino_types)}")
        
        # Create residue positions array
        pos_ca = np.full((len(amino_types), 3), np.nan)
        
        # Handle the case with no CA atoms
        if np.sum(mask_ca) == 0:
            print("WARNING: No CA atoms found! Using first atom of each residue.")
            # Alternative: Use the first atom of each residue
            first_atoms = {}
            for i, res_idx in enumerate(atom_amino_id):
                if res_idx not in first_atoms:
                    first_atoms[res_idx] = i
            
            for res_idx, atom_idx in first_atoms.items():
                if res_idx < len(amino_types):
                    pos_ca[res_idx] = atom_pos[atom_idx]
        else:
            # Use identified CA atoms
            ca_indices = atom_amino_id[mask_ca]
            if len(ca_indices) == 0:
                print("WARNING: No CA atoms mapped to residues!")
                pos_ca = np.zeros((len(amino_types), 3))
            else:
                valid_indices = ca_indices < len(amino_types)
                if not np.all(valid_indices):
                    print(f"WARNING: Some CA indices out of bounds: {ca_indices[~valid_indices]}")
                    ca_indices = ca_indices[valid_indices]
                    mask_ca_filtered = np.zeros_like(mask_ca)
                    mask_ca_filtered[np.where(mask_ca)[0][valid_indices]] = True
                    mask_ca = mask_ca_filtered
                
                pos_ca[ca_indices] = atom_pos[mask_ca]
        
        # Check for NaN values
        nan_count = np.isnan(pos_ca).any(axis=1).sum()
        if nan_count > 0:
            print(f"WARNING: {nan_count} residues have NaN coordinates!")
            # Replace NaNs with zeros to avoid errors
            pos_ca = np.nan_to_num(pos_ca)
        
        pos_ca = torch.FloatTensor(pos_ca)
        
        # Edge creation
        edge_index, edge_dists = atom_coords_to_edges(pos_ca, edge_dist_cutoff=12)
        
        # Create the graph
        res_graph = ResidueGraph(node_pos=pos_ca,
                                edge_index=edge_index,
                                edge_attr=edge_dists,
                                node_names=amino_ids)
        
        # Add features
        res_graph.features.add_named_oh_features('amino_types', amino_types, 21)
        res_graph.features.add_named_oh_features('sse', res_sse, nclasses=8)
        
        hphob = np.asarray([res_type_to_hphob.get(amino_type, 0) for amino_type in amino_types], dtype=np.float32)
        res_graph.features.add_named_features('hphobs', hphob)
        
        if self.add_esm and pdb_path is not None:
            esm_embed = get_esm_embedding_single(pdb_path, esm_path)
            res_graph.features.add_named_features('esm_embed', esm_embed)
        
        if self.add_pronet:
            pfc = PronetFeaturesComputer()
            pronet_features = pfc.get_pronet_features(amino_types, atom_amino_id, atom_names, atom_pos)
            res_graph.features.add_misc_features("pronet_features", pronet_features)
        
        return res_graph

    def pdb_to_resgraph(self, pdb_path, esm_path=None):
        arrays = parse_pdb_path(pdb_path)
        return self.arrays_to_resgraph(arrays=arrays, pdb_path=pdb_path, esm_path=esm_path)


if __name__ == "__main__":
    pass
    pdb = "../../data/example_files/4kt3.pdb"
    resgraph_path = "../../data/example_files/4kt3_resgraph.pt"
    residue_graph_builder = ResidueGraphBuilder()
    residue_graph = residue_graph_builder.pdb_to_resgraph(pdb)
    torch.save(residue_graph, resgraph_path)
    residue_graph = torch.load(resgraph_path)
    a = 1

    pdb = "./code/MasifLigand/raw_data_MasifLigand/pdb/2V1O_ACBEDF.pdb"
    residue_graph_builder = ResidueGraphBuilder(add_esm=True)
    residue_graph = residue_graph_builder.pdb_to_resgraph(pdb)
    print(residue_graph.features['named_features'])
