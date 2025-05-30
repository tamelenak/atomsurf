import os
import sys

from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
from pathlib import Path
import scipy.spatial as ss
from subprocess import Popen, PIPE
import shutil
import torch
from torch_geometric.utils import to_undirected
from Bio.PDB.DSSP import DSSP

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

# atom type label for one-hot-encoding
atom_type_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'P': 6, 'Cl': 7, 'Se': 8,
                  'Br': 9, 'I': 10, 'UNK': 11}
# residue type label for one-hot-encoding
res_type_dict = {
    'ALA': 0, 'GLY': 1, 'SER': 2, 'THR': 3, 'LEU': 4, 'ILE': 5, 'VAL': 6, 'ASN': 7, 'GLN': 8, 'ARG': 9, 'HIS': 10,
    'TRP': 11, 'PHE': 12, 'TYR': 13, 'GLU': 14, 'ASP': 15, 'LYS': 16, 'PRO': 17, 'CYS': 18, 'MET': 19, 'UNK': 20, }

protein_letters_1to3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "X": "Unk"
}

protein_letters_3to1 = {value.upper(): key for key, value in protein_letters_1to3.items()}

res_type_idx_to_1 = {
    idx: protein_letters_3to1[res_type] for res_type, idx in res_type_dict.items()
}

# Kyte Doolittle scale for hydrophobicity
hydrophob_dict = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7,
    'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5,
    'LYS': -3.9, 'ARG': -4.5, 'UNK': 0.0,
}

res_type_to_hphob = {
    idx: hydrophob_dict[res_type] for res_type, idx in res_type_dict.items()
}
SSE_type_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}


def quick_pdb_to_seq(pdb_path):
    """
    No need for extensive parsing to only retrieve the sequence
    :param pdb_path:
    :return:
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("toto", pdb_path)
    amino_types = []  # size: (n_amino,)
    for residue in structure.get_residues():
        # HETATM
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname()
        if resname.upper() not in res_type_dict:
            resname = 'UNK'
        resname = res_type_dict[resname.upper()]
        amino_types.append(resname)
    amino_types = np.asarray(amino_types, dtype=np.int32)
    return amino_types


def extract_chains(input_pdb, output_pdb, chains_to_extract, recompute=False, verbose=False):
    if os.path.exists(output_pdb) and not recompute:
        return
    # Initialize the PDB parser and structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)

    # Create a new structure object to store the extracted chains
    new_structure = PDB.Structure.Structure('extracted_chains')

    for model in structure:
        new_model = PDB.Model.Model(model.id)
        for chain in model:
            if chain.id in chains_to_extract:
                new_model.add(chain)
        if len(new_model):
            new_structure.add(new_model)

    # Save the new structure with the selected chains
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    if verbose:
        print(f"Chains {', '.join(chains_to_extract)} extracted and saved to {output_pdb}")


def parse_pdb_path(pdb_path, use_pqr=True):
    if use_pqr:
        pdb2pqr_bin = shutil.which('pdb2pqr')
        if pdb2pqr_bin is None:
            raise RuntimeError('pdb2pqr executable not found')

        pdb_path = Path(pdb_path)
        # process pqr
        out_dir = pdb_path.parent
        pdb_id = pdb_path.stem
        pqr_path = Path(out_dir / f'{pdb_id}.pqr')
        pqr_log_path = Path(out_dir / f'{pdb_id}.log')
        if not pqr_path.exists():
            cmd = [pdb2pqr_bin, '--ff=AMBER', str(pdb_path), str(pqr_path), '--keep-chain']
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = proc.communicate()
            err = stderr.decode('utf-8').strip('\n')
            if 'CRITICAL' in err:
                print(f'{pdb_id} pdb2pqr failed', flush=True)
                return None, None, None, None, None, None, None, None, None, None, None
        parser = PDBParser(QUIET=True, is_pqr=True)
        structure = parser.get_structure("toto", pqr_path)
    else:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("toto", pdb_path)

    amino_types = []  # size: (n_amino,)
    atom_chain_id = []  # size:(n_atom,)
    atom_amino_id = []  # size: (n_atom,)
    atom_names = []  # size: (n_atom,)
    atom_types = []  # size: (n_atom,)
    atom_pos = []  # size: (n_atom,3)
    atom_charge = []  # size: (n_atom,1)
    atom_radius = []  # size: (n_atom,1)
    amino_ids = []  # size: (n_amino,) 1 letter AA type + seq position in PDB file, not used as features
    atom_ids = []  # size: (n_atom,) 1 letter AA type + seq position in PDB + atom name, not used as features
    res_id = 0
    for residue in structure.get_residues():
        for atom in residue.get_atoms():
            # Add occupancy to write as pdb
            atom.set_occupancy(1.0)
            atom.set_bfactor(1.0)

        # HETATM
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname()
        res_pdb_pos = residue.get_id()[1]
        # resname = protein_letters_3to1[resname.title()]
        if resname.upper() not in res_type_dict:
            resname = 'UNK'
        resname = res_type_dict[resname.upper()]
        res_unique_id = f"{residue.full_id[2]}:{res_type_idx_to_1[resname]}{str(res_pdb_pos)}"  # chain:AApos e.g. B:I26
        amino_types.append(resname)
        amino_ids.append(res_unique_id)

        for atom in residue.get_atoms():
            # Skip H
            element = atom.element
            if atom.get_name().startswith("H"):
                continue
            if not element in atom_type_dict:
                element = 'UNK'
            atom_chain_id.append(residue.full_id[2])
            atom_types.append(atom_type_dict[element])
            atom_names.append(atom.get_name())
            atom_pos.append(atom.get_coord())
            atom_amino_id.append(res_id)
            atom_ids.append(res_unique_id + "_" + atom.get_id())
            if use_pqr:
                atom_charge.append(atom.get_charge())
                atom_radius.append(atom.get_radius())

        res_id += 1
    amino_types = np.asarray(amino_types, dtype=np.int32)
    atom_chain_id = np.asarray(atom_chain_id)
    atom_amino_id = np.asarray(atom_amino_id, dtype=np.int32)
    atom_names = np.asarray(atom_names)
    atom_types = np.asarray(atom_types, dtype=np.int32)
    atom_pos = np.asarray(atom_pos, dtype=np.float32)
    atom_charge = np.asarray(atom_charge, dtype=np.float32) if use_pqr else None
    atom_radius = np.asarray(atom_radius, dtype=np.float32) if use_pqr else None
    amino_ids = np.asarray(amino_ids, dtype=object)
    atom_ids = np.asarray(atom_ids, dtype=object)

    # We need to dump this adapted pdb with new coordinates and missing atoms
    if use_pqr:
        from Bio.PDB.PDBIO import PDBIO
        io = PDBIO()
        io.set_structure(structure)
        pqrpdbpath = str(pqr_path) + 'pdb'
        io.save(pqrpdbpath)
        p = PDBParser(QUIET=True)
        structure = p.get_structure("test", pqrpdbpath)

    # process DSSP, if installed and not buggy
    try:
        dssp = DSSP(structure[0], pqrpdbpath if use_pqr else pdb_path, file_type="PDB")
        # Make sure DSSP is consistent with residue number in pdb. Map to unknown if no SSE found for a residue
        res_sse_list = [len(SSE_type_dict) for _ in range(len(amino_types))]

        shift = 0
        dssp_keys = list(dssp.keys())
        for i, amino_id in enumerate(amino_ids):
            if i - shift >= len(dssp_keys):
                break
            key = dssp_keys[i - shift]
            res_id = f"{key[0]}:{dssp[key][1]}{key[1][1]}"  # chain:res_type_res_pos
            if res_id == amino_id:
                res_sse_list[i] = SSE_type_dict[dssp[key][2]]
            else:
                shift += 1
        res_sse = np.array(res_sse_list)

    except Exception as e:
        res_sse = np.full_like(amino_types, len(SSE_type_dict), dtype=np.int64)

    if use_pqr:
        os.remove(pqr_path)
        os.remove(pqr_log_path)
        os.remove(pqrpdbpath)
    return amino_types, atom_chain_id, atom_amino_id, atom_names, atom_types, atom_pos, atom_charge, atom_radius, res_sse, amino_ids, atom_ids


def atom_coords_to_edges(node_pos, edge_dist_cutoff=4.5):
    r"""
    Turn nodes position into neighbors graph.
    """
    # from torch_geometric.nn import radius_graph
    # edges = radius_graph(node_pos, r=edge_dist_cutoff, batch=torch.zeros(len(node_pos),dtype=int), max_num_neighbors=48)
    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)
    # print(f"time to pre_dist : {time.time() - t0}")

    # t0 = time.time()
    node_a = node_pos[edges[0, :]]
    node_b = node_pos[edges[1, :]]
    with torch.no_grad():
        my_edge_weights_torch = 1 / (np.linalg.norm(node_a - node_b, axis=1) + 1e-5)
    return edges, my_edge_weights_torch


if __name__ == "__main__":
    pass
