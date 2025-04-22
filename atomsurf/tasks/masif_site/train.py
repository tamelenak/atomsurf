# std
import sys
from pathlib import Path
import torch
# 3p
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from atomsurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger
from pl_model import MasifSiteModule
from data_loader import MasifSiteDataModule
import warnings 
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

# Hardcode configuration parameters
seed = 2024
run_name = "default"
data_dir = "../../../data/masif_site/"
out_dir = "../../../outputs/masif_site/out_dir"
log_dir = "./"
test_freq = 5
epochs = 200
device = 0
path_model = "version_x/checkpoints/last.ckpt"
min_batch_size = 2
use_wandb = False

# Loader parameters
num_workers = 2
batch_size = 16
pin_memory = False
prefetch_factor = 2
shuffle = True

# Training parameters
save_top_k = 2
early_stopping_patience = 100
accumulate_grad_batches = 1
val_check_interval = 1.0
check_val_every_n_epoch = 1
limit_train_batches = 1.0
limit_val_batches = 1.0
limit_test_batches = 1.0
gradient_clip_val = 1.0
deterministic = False
max_steps = -1
auto_lr_find = False
log_every_n_steps = 50
detect_anomaly = False
overfit_batches = 0
to_monitor = "auroc/val"

from types import SimpleNamespace

# Create configuration objects using SimpleNamespace
cfg_surface = SimpleNamespace(
    use_surfaces=True,
    feat_keys='all',
    oh_keys='all',
    gdf_expand=True,
    data_dir=f"{data_dir}/surfaces_0.1_False",
    data_name='surfaces_0.1_False'
)

cfg_graph = SimpleNamespace(
    use_graphs=True,
    feat_keys='all',
    oh_keys='all',
    esm_dir=f"{data_dir}/01-benchmark_esm_embs",
    use_esm=False,
    data_dir=f"{data_dir}/rgraph"
)

loader_cfg = SimpleNamespace(
    num_workers=num_workers,
    batch_size=batch_size,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    shuffle=shuffle
)

# Combine into a single configuration
cfg = SimpleNamespace(
    cfg_surface=cfg_surface,
    cfg_graph=cfg_graph,
    loader=loader_cfg,
    data_dir=data_dir
)

def main():
    command = f"python3 {' '.join(sys.argv)}"
    
    pl.seed_everything(seed, workers=True)

    # init datamodule
    datamodule = MasifSiteDataModule(cfg=cfg)

    # init model
    model = MasifSiteModule()

    # init logger
    version = TensorBoardLogger(save_dir=log_dir).version
    version_name = f"version_{version}_{run_name}"
    tb_logger = TensorBoardLogger(save_dir=log_dir, version=version_name)
    loggers = [tb_logger]

    if use_wandb:
        add_wandb_logger(loggers, projectname="masif_site")

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{accuracy_balanced/val:.2f}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor=to_monitor,
        mode="max",
        save_last=True,
        save_top_k=save_top_k,
        verbose=False,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor=to_monitor,
                                                     patience=early_stopping_patience,
                                                     mode='max')
    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, CommandLoggerCallback(command)]

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [device]}
    else:
        params = {}

    # init trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        max_steps=max_steps,
        gradient_clip_val=gradient_clip_val,
        detect_anomaly=detect_anomaly,
        overfit_batches=overfit_batches,
        **params,
    )

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
