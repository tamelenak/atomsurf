import torch
import torch.nn.functional as F

from atomsurf.tasks.masif_site.model import MasifSiteNet


class GraphOnlyMasifSiteWrapper(torch.nn.Module):
    """
    Wrapper around MasifSiteNet that uses GCN-only encoder blocks
    and maps graph predictions to surface points for loss computation.
    
    This enables fair comparison between hybrid (surface+graph) vs graph-only
    approaches using the same GCN architecture.
    """
    
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.base_model = MasifSiteNet(cfg_encoder, cfg_head)
    
    def forward(self, batch):
        """
        Forward pass through GCN-only encoder, then map to surface
        """
        # The encoder will only process graph (surface_encoder=None in gcnonly blocks)
        # but we still need to provide surface for the mapping
        surface, graph = self.base_model.encoder(graph=batch.graph, surface=batch.surface)
        
        # Graph has been processed with GCN, surface should be unchanged (since surface_encoder=None)
        # Now we need to map graph features to surface points
        if surface is not None and graph is not None:
            # Map graph node features to surface points using nearest neighbors
            surface_features = self._map_graph_to_surface(
                graph.x, graph.node_pos, surface.pos, surface.batch, graph.batch
            )
            surface.x = surface_features
        
        # Apply the top network to surface features (mapped from graph)
        surface.x = self.base_model.top_net(surface.x)
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