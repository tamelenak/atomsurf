import csv
import pytorch_lightning as pl
import torch

class BatchStatsLogger(pl.Callback):
    def __init__(self, filename="batch_stats.csv"):
        self.filename = filename
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch_idx", "loss", "accuracy", "vertex_count", "node_count", "batch_size", "protein_names"
            ])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        epoch = trainer.current_epoch
        loss = outputs["loss"].item() if isinstance(outputs, dict) and "loss" in outputs else outputs
        accuracy = outputs["accuracy"] if isinstance(outputs, dict) and "accuracy" in outputs else None

        # Vertex count: total number of points/vertices in the batch
        if hasattr(batch, "label"):
            try:
                concatenated_labels = torch.cat(batch.label) if isinstance(batch.label, (list, tuple)) else batch.label
                vertex_count = len(concatenated_labels)
            except Exception:
                vertex_count = 0
        else:
            vertex_count = 0

        node_count = float(batch.graph.node_len.float().sum().item()) if hasattr(batch, "graph") and hasattr(batch.graph, "node_len") else 0
        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 0

        # Extract protein names from the batch
        protein_names = getattr(batch, 'protein_name', [])
        protein_names_str = ";".join(protein_names) if protein_names else ""

        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch_idx, loss, accuracy, vertex_count, node_count, batch_size, protein_names_str
            ])