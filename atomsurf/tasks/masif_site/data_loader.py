import os
import sys

import numpy as np
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim
#from uniform_batch_sampler import UniformBatchSampler


class MasifSiteDataset(Dataset):
    def __init__(self, systems, surface_builder, graph_builder, verbose=False):
        self.systems = systems
        self.surface_builder = surface_builder
        self.graph_builder = graph_builder
        self.verbose = verbose

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        protein_name = self.systems[idx]
        surface = self.surface_builder.load(protein_name)
        graph = self.graph_builder.load(protein_name)
        if surface is None or graph is None:
            if self.verbose:
                print(f"Failed to load: {protein_name} (surface: {surface is None}, graph: {graph is None})")
            return None
        item = Data(surface=surface, graph=graph, label=surface.iface_labels if hasattr(surface, 'iface_labels') else None, protein_name=protein_name)
        assert item is not None, f"Got None at idx {idx}"
        return item


def collate_fn(batch):
    # Filter out None values (failed loads)
    batch_before = batch.copy()
    batch = [x for x in batch if x is not None]
    if len(batch) != len(batch_before):
        print(f"Batch size before filtering: {len(batch)}")
        print(f"Batch size after filtering: {len(batch)}")
    if len(batch) == 0:
        print("Warning: Empty batch after filtering!")
        # Create a minimal empty batch to avoid errors
        return AtomBatch()
    
    assert all(x is not None for x in batch), "Found None in batch!"
    return AtomBatch.from_data_list(batch)

class MasifSiteDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.surface_loader = SurfaceLoader(cfg.cfg_surface)
        self.graph_loader = GraphLoader(cfg.cfg_graph)
        
        # Set verbose mode (default to False if not specified in config)
        self.verbose = cfg.verbose if hasattr(cfg, 'verbose') else False

        # Get the right systems
        # script_dir = os.path.dirname(os.path.realpath(__file__))
        # masif_site_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')
        masif_site_data_dir= cfg.data_dir
        train_systems_list = os.path.join(masif_site_data_dir, 'splits/train_list.txt')
        trainval_sys = [name.strip() for name in open(train_systems_list, 'r').readlines()]
        np.random.shuffle(trainval_sys)
        trainval_cut = int(0.9 * len(trainval_sys))
        self.train_sys = trainval_sys[:trainval_cut]
        self.val_sys = trainval_sys[trainval_cut:]
        test_systems_list = os.path.join(masif_site_data_dir, 'splits/test_list.txt')
        self.test_sys = [name.strip() for name in open(test_systems_list, 'r').readlines()]

        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': collate_fn}

        dataset_temp = MasifSiteDataset(self.train_sys, self.surface_loader, self.graph_loader, verbose=False)
        update_model_input_dim(cfg, dataset_temp=dataset_temp)

    def train_dataloader(self):
        dataset = MasifSiteDataset(self.train_sys, self.surface_loader, self.graph_loader, verbose=self.verbose)
        print(f"Training dataset size: {len(self.train_sys)}")
        print(f"Batch size: {self.cfg.loader.batch_size}")
        
        # Create the uniform batch sampler
        #sampler = UniformBatchSampler(
        #    dataset=dataset,
        #    batch_size=self.cfg.loader.batch_size,
        #    num_bins=5,  # You can adjust this based on your dataset size
        #    shuffle=True,
        #    drop_last=False
        #)
        #sampler = RandomSampler(dataset)
        
        # Remove batch_size and shuffle from loader_args since we're using a sampler
        #loader_args = self.loader_args.copy()
        #loader_args.pop('batch_size', None)
        #loader_args.pop('shuffle', None)
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = MasifSiteDataset(self.val_sys, self.surface_loader, self.graph_loader, verbose=self.verbose)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = MasifSiteDataset(self.test_sys, self.surface_loader, self.graph_loader, verbose=self.verbose)
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == '__main__':
    import omegaconf

    np.random.seed(42)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_site_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')

    # SURFACE
    cfg_surface = omegaconf.DictConfig({})
    cfg_surface.use_surfaces = True
    cfg_surface.feat_keys = 'all'
    cfg_surface.oh_keys = 'all'
    cfg_surface.gdf_expand = True
    # cfg_surface.data_dir = os.path.join(masif_site_data_dir, 'surfaces_1.0_False')
    cfg_surface.data_dir = os.path.join(masif_site_data_dir, 'surfaces_0.1_False')
    surface_loader = SurfaceLoader(cfg_surface)

    # GRAPHS
    cfg_graph = omegaconf.DictConfig({})
    cfg_graph.use_graphs = True
    cfg_graph.feat_keys = 'all'
    cfg_graph.oh_keys = 'all'
    cfg_graph.esm_dir = os.path.join(masif_site_data_dir, '01-benchmark_esm_embs')
    cfg_graph.use_esm = True
    cfg_graph.data_dir = os.path.join(masif_site_data_dir, 'rgraph')
    # cfg_graph.data_dir = os.path.join(masif_site_data_dir, 'agraph')
    graph_loader = GraphLoader(cfg_graph)

    # test_systems_list = os.path.join(masif_site_data_dir, 'test_list.txt')
    # test_sys = [name.strip() for name in open(test_systems_list, 'r').readlines()]
    # dataset = MasifSiteDataset(test_sys, surface_loader, graph_loader, verbose=False)
    # a = dataset[0]

    loader_cfg = omegaconf.DictConfig({"num_workers": 0,
                                       "batch_size": 4,
                                       "pin_memory": False,
                                       "prefetch_factor": 2,
                                       "shuffle": False})
    simili_cfg = omegaconf.DictConfig({"cfg_surface": cfg_surface, "cfg_graph": cfg_graph, "loader": loader_cfg})

    # We need to add a pretend encoder there because input size is updated automatically based on features
    encoder_cfg = omegaconf.DictConfig({"blocks": [{"kwargs": {}}]})
    simili_cfg.encoder = encoder_cfg

    datamodule = MasifSiteDataModule(cfg=simili_cfg)
    loader = datamodule.train_dataloader()
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    for i, batch in enumerate(loader):
        pass
        a = batch.protein_name
        batch = batch.to(device)
        torch.cuda.synchronize()

        if i > 400:
            break
    time_elapsed = time.time() - t0
    # Mean time elasped without sending to device : 0.02
    print('total : ', time_elapsed, 'mean time_loop', time_elapsed / 400)
