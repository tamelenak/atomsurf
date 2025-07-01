from pathlib import Path

# 3p
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import wandb


def add_wandb_logger(loggers, projectname,runname):
    # init logger
    wandb.init(reinit=True, entity='vincent-mallet-cri-lpi')
    wand_id = wandb.util.generate_id()
    tb_logger = loggers[-1]
    run_name = f"{Path(tb_logger.log_dir).stem}"
    tags = []
    Path(tb_logger.log_dir).absolute().mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project=projectname, name=run_name, tags=tags,
                               version=Path(tb_logger.log_dir).stem, id=wand_id,
                               save_dir=tb_logger.log_dir, log_model=False)
    loggers += [wandb_logger]


class CommandLoggerCallback(Callback):
    def __init__(self, command):
        self.command = command

    def setup(self, trainer, pl_module, stage):
        tensorboard = pl_module.loggers[0].experiment
        tensorboard.add_text("Command", self.command)


class ExperimentTrackerCallback(Callback):
    def __init__(self, encoder_name, log_dir, run_name):
        import csv
        import os
        self.encoder_name = encoder_name
        self.log_dir = log_dir
        self.run_name = run_name
        self.csv = csv
        self.os = os

    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':
            log_path = self.os.path.join(self.log_dir, 'experiment_tracker.csv')
            file_exists = self.os.path.isfile(log_path)
            with open(log_path, 'a', newline='') as csvfile:
                fieldnames = ['run_name', 'encoder_name']
                writer = self.csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'run_name': self.run_name, 'encoder_name': self.encoder_name})
