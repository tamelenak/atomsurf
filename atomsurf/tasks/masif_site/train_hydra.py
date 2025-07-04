# std
import sys
from pathlib import Path
# 3p
import hydra
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

# from batch_stats_logger import BatchStatsLogger

from atomsurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger
from pl_model import MasifSiteModule
from data_loader import MasifSiteDataModule
import warnings 
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init datamodule
    datamodule = MasifSiteDataModule(cfg)

    # init model
    model = MasifSiteModule(cfg)

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    if cfg.use_wandb:
        add_wandb_logger(loggers, projectname="masif_site")

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{accuracy_balanced}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor=cfg.train.to_monitor,
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=False,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor=cfg.train.to_monitor,
                                                     patience=cfg.train.early_stoping_patience,
                                                     mode='max')
    
    callbacks = [
        lr_logger, 
        checkpoint_callback, 
        early_stop_callback, 
        CommandLoggerCallback(command),
        # BatchStatsLogger("batch_stats.csv")
    ]

    # Use the accelerator from config, defaulting to CPU if not specified
    params = {
        "accelerator": cfg.accelerator if hasattr(cfg, 'accelerator') else ("gpu" if torch.cuda.is_available() else "cpu")
    }
    
    # Only add devices and strategy if using GPU
    if params["accelerator"] == "gpu":
        if hasattr(cfg, 'devices'):
            params["devices"] = cfg.devices
        if hasattr(cfg, 'strategy'):
            params["strategy"] = cfg.strategy

    # init trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        # epochs, batch size and when to val
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        max_steps=cfg.train.max_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        detect_anomaly=cfg.train.detect_anomaly,
        overfit_batches=cfg.train.overfit_batches,
        **params,
    )

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main() 