import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifSiteNet(torch.nn.Module):
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.hparams_head = cfg_head
        self.hparams_encoder = cfg_encoder
        self.encoder = ProteinEncoder(cfg_encoder)

        self.top_net = nn.Sequential(*[
            nn.Linear(cfg_head.encoded_dims, cfg_head.encoded_dims),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(cfg_head.encoded_dims),
            #nn.SiLU(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(cfg_head.encoded_dims, out_features=cfg_head.output_dims),
            #nn.Sigmoid()
            nn.LeakyReLU(negative_slope=0.2)
        ])

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)
        
        # Check if surface has been encoded (for graph-only encoders, it won't be)
        if surface.x.shape[1] != self.hparams_head.encoded_dims:
            # Surface hasn't been encoded, map graph features to surface points
            surface.x = self._map_graph_to_surface(
                graph_features=graph.x,
                graph_pos=graph.node_pos,
                surface_pos=batch.surface.verts,
                surface_batch=batch.surface.batch,
                graph_batch=graph.batch
            )
        
        surface.x = self.top_net(surface.x)
        return surface
    
    def _map_graph_to_surface(self, graph_features, graph_pos, surface_pos, surface_batch, graph_batch):
        """
        Map graph node features to surface points using nearest neighbors
        Properly handles batched data by matching batch indices
        """
        device = graph_features.device
        mapped_features = torch.zeros(
            surface_pos.shape[0], graph_features.shape[1], 
            device=device, dtype=graph_features.dtype
        )
        
        # Process each protein in the batch separately
        batch_ids = torch.unique(surface_batch)
        
        for batch_id in batch_ids:
            # Get indices for this protein in surface and graph
            surface_mask = surface_batch == batch_id
            graph_mask = graph_batch == batch_id
            
            surface_points_protein = surface_pos[surface_mask]
            graph_points_protein = graph_pos[graph_mask]
            graph_features_protein = graph_features[graph_mask]
            
            # Only proceed if we have both surface and graph data for this protein
            if len(surface_points_protein) > 0 and len(graph_points_protein) > 0:
                # Compute distances between surface points and graph nodes for this protein
                distances = torch.cdist(surface_points_protein, graph_points_protein)
                nearest_indices = torch.argmin(distances, dim=1)
                
                # Assign features from nearest graph nodes
                mapped_features[surface_mask] = graph_features_protein[nearest_indices]
        
        return mapped_features
