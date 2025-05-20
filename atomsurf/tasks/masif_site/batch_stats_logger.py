import csv
import pytorch_lightning as pl

class BatchStatsLogger(pl.Callback):
    def __init__(self, filename="batch_stats.csv"):
        self.filename = filename
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch_idx", "loss", "accuracy", "vertex_count", "node_count", "batch_size"
            ])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        epoch = trainer.current_epoch
        loss = outputs["loss"].item() if isinstance(outputs, dict) and "loss" in outputs else outputs
        accuracy = outputs["accuracy"] if isinstance(outputs, dict) and "accuracy" in outputs else None
        try:
            data_list = batch.to_data_list()
        except Exception:
            data_list = []
        vertex_count = sum(len(d.surface.verts) for d in data_list if hasattr(d.surface, "verts"))
        node_count = sum(len(d.graph.x) for d in data_list if hasattr(d.graph, "x"))
        batch_size = len(data_list)
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch_idx, loss, accuracy, vertex_count, node_count, batch_size
            ])