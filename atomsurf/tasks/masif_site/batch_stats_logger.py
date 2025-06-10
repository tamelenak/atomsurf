import csv
import pytorch_lightning as pl
import torch
from queue import Queue
from threading import Thread
import time

class BatchStatsLogger(pl.Callback):
    def __init__(self, filename="batch_stats.csv"):
        self.filename = filename
        self.queue = Queue()
        self.writing = True
        
        # Initialize CSV file
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch_idx", "loss", "accuracy", "vertex_count", "node_count", "batch_size", "protein_names"
            ])
        
        # Start writer thread
        self.writer_thread = Thread(target=self._write_worker, daemon=True)
        self.writer_thread.start()

    def _write_worker(self):
        while self.writing:
            try:
                # Get data from queue with timeout
                data = self.queue.get(timeout=1.0)
                with open(self.filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                self.queue.task_done()
            except:
                # Timeout or other error, just continue
                continue

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

        # Add data to queue instead of writing directly
        self.queue.put([
            epoch, batch_idx, loss, accuracy, vertex_count, node_count, batch_size, protein_names_str
        ])

    def on_train_end(self, trainer, pl_module):
        # Stop writer thread
        self.writing = False
        self.writer_thread.join()