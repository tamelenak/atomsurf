import torch
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm

from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifLigandNet(torch.nn.Module):
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.hparams_head = cfg_head
        self.hparams_encoder = cfg_encoder
        self.encoder = ProteinEncoder(cfg_encoder)

        norm_type = getattr(cfg_head, 'norm_type', 'batch').lower()
        
        if norm_type == 'layer':
            norm_layer = nn.LayerNorm(cfg_head.encoded_dims)
        elif norm_type == 'batch':
            norm_layer = nn.BatchNorm1d(cfg_head.encoded_dims)
        elif norm_type == 'graph':
            norm_layer = GraphNorm(cfg_head.encoded_dims)
        else:
            raise ValueError(f"norm_type {norm_type} not supported")

        self.top_net = nn.Sequential(*[
            nn.Linear(cfg_head.encoded_dims, cfg_head.encoded_dims),
            nn.Dropout(p=0.1),
            norm_layer,
            nn.SiLU(),
            nn.Linear(cfg_head.encoded_dims, out_features=cfg_head.output_dims)
        ])

    def pool_lig(self, pos, processed, lig_coords):
        # find nearest neighbors between last layers encodings and position of the ligand
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
        return processed[min_indices]

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)

        # Decide which modality to pool on (default: surface). Allows graph-only comparisons.
        pool_on = getattr(self.hparams_encoder, 'pool_on', 'surface')

        if pool_on == 'graph':
            if graph is None:
                raise ValueError("Graph pooling requested but graph is None. Ensure graph encoder/inputs are enabled.")
            pos_and_x = [(g.node_pos, g.x) for g in graph.to_data_list()]
        else:
            if surface is None:
                raise ValueError("Surface pooling requested but surface is None. Ensure surface encoder/inputs are enabled.")
            pos_and_x = [(surf.verts, surf.x) for surf in surface.to_data_list()]

        pockets_embs = []
        for (pos, x), lig_coord in zip(pos_and_x, batch.lig_coord):
            selected = self.pool_lig(pos, x, lig_coord)
            pocket_emb = torch.mean(selected, dim=-2)
            pockets_embs.append(pocket_emb)
        x = torch.stack(pockets_embs)
        x = self.top_net(x)
        return x
