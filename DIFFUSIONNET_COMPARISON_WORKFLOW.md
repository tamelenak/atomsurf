# DiffusionNet++ vs Traditional DiffusionNet Comparison Workflow

## Goal
Compare DiffusionNet++ (batched implementation) with Traditional DiffusionNet (non-batched) at batch_size=1 to verify DiffusionNet++ correctness.

## Background
- **Traditional DiffusionNet**: Original implementation from nmwsharp/diffusion-net (non-batched, single samples)
- **DiffusionNet++**: Enhanced version from pvnieo/diffusion-net-plus (supports batching with PyTorch Geometric)
- **atomsurf**: Uses DiffusionNet++ by default with PyG Data objects

## Setup

### 1. Conda Environments

#### atomsurf (DiffusionNet++)
```bash
conda activate atomsurf
# Uses diffusion-net-plus installed from pip
```

#### atomsurf_traditional (Traditional DiffusionNet)
```bash
conda activate atomsurf_traditional
# Uses local clone of traditional diffusion-net via PYTHONPATH
```

### 2. Traditional DiffusionNet Setup
Located at: `/root/atomsurf/diffusion-net/` (cloned from https://github.com/nmwsharp/diffusion-net.git)

**Modifications made** (in `/root/atomsurf/diffusion-net/src/diffusion_net/layers.py`):
- Created `DiffusionNetBlockPyGWrapper` class to bridge PyG Data interface with traditional DiffusionNet
- Wrapper handles:
  - PyG Data object extraction
  - Sparse tensor → dense tensor conversion
  - Mass matrix diagonal extraction
  - Batch dimension management

**Key wrapper features:**
```python
class DiffusionNetBlockPyGWrapper(nn.Module):
    # Accepts atomsurf config params (use_bn, use_layernorm, init_time, init_std, n_layers)
    # Wraps original DiffusionNetBlock
    # Converts PyG Data → traditional format
    # Handles batch dimension addition/removal
```

### 3. Configuration Files Created

#### `/root/atomsurf/atomsurf/tasks/shared_conf/blocks_zoo.yaml`
Added:
```yaml
diffusion_net_traditional:
  _target_: diffusion_net.DiffusionNetBlockPyGWrapper
  C_width: ${model_hdim}
  dropout: ${model_dropout}
  use_bn: ${eval:'not ${use_layernorm}'}
  use_layernorm: ${use_layernorm}
  init_time: ${diff_time_mean}
  init_std: ${diff_time_std}

diffonly_traditional:
  _target_: atomsurf.networks.ProteinEncoderBlock
  surface_encoder: ${diffusion_net_traditional}
  graph_encoder: None
  message_passing: None
```

#### `/root/atomsurf/atomsurf/tasks/shared_conf/encoder/surf_only_traditional.yaml`
```yaml
name: surf_only_traditional

blocks:
  - ${input_encoder_gvp}
  - ${diffonly_traditional}
```

## Running the Comparison

### Training 1: Traditional DiffusionNet (batch_size=1)
```bash
tmux new -s trad_diffnet

# Inside tmux:
source /root/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf_traditional
cd /root/atomsurf/atomsurf/tasks/masif_ligand

PYTHONPATH=/root/atomsurf/diffusion-net/src:$PYTHONPATH python3 train.py \
  encoder=surf_only_traditional.yaml \
  optimizer.lr=0.0001 \
  scheduler=reduce_lr_on_plateau \
  epochs=100 \
  loader.batch_size=1 \
  loader.num_workers=16 \
  diffusion_net.use_bn=false \
  diffusion_net.use_layernorm=true \
  diffusion_net.init_time=2.0 \
  diffusion_net.init_std=2.0 \
  train.save_top_k=5 \
  train.early_stoping_patience=500 \
  min_batch_size=1 \
  run_name=traditional_diffnet_bs1 \
  device=2

# Detach: tmux detach
```

### Training 2: DiffusionNet++ (batch_size=1)
```bash
tmux new -s plus_diffnet

# Inside tmux:
source /root/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf
cd /root/atomsurf/atomsurf/tasks/masif_ligand

python3 train.py \
  encoder=surf_only.yaml \
  optimizer.lr=0.0001 \
  scheduler=reduce_lr_on_plateau \
  epochs=100 \
  loader.batch_size=1 \
  loader.num_workers=16 \
  diffusion_net.use_bn=false \
  diffusion_net.use_layernorm=true \
  diffusion_net.init_time=2.0 \
  diffusion_net.init_std=2.0 \
  train.save_top_k=5 \
  train.early_stoping_patience=500 \
  min_batch_size=1 \
  run_name=plus_diffnet_bs1 \
  device=3

# Detach: tmux detach
```

## Monitoring

### Check running sessions
```bash
tmux ls
```

### Attach to view progress
```bash
tmux a -t trad_diffnet    # Traditional DiffusionNet
tmux a -t plus_diffnet    # DiffusionNet++
```

### Check processes
```bash
ps aux | grep python3 | grep train.py | grep diffnet
```

### Detach all sessions
```bash
tmux list-sessions | grep attached | cut -d: -f1 | xargs -I {} tmux detach-client -s {}
```

## Results Location

Training outputs saved to:
```
/root/atomsurf/atomsurf/tasks/masif_ligand/outputs/
├── traditional_diffnet_bs1/
│   ├── checkpoints/
│   ├── logs/
│   └── ...
└── plus_diffnet_bs1/
    ├── checkpoints/
    ├── logs/
    └── ...
```

## Comparison Metrics

Compare the two runs on:
- **Loss convergence**: Should be similar
- **Validation accuracy**: Should be comparable
- **Training speed**: May differ (batched vs non-batched processing)
- **Memory usage**: May differ

## Key Differences

| Aspect | Traditional DiffusionNet | DiffusionNet++ |
|--------|--------------------------|----------------|
| Batching | No batching (processes single samples) | Supports batching |
| Tensor format | Dense tensors | PyG sparse tensors |
| Batch dimension | Internally adds/removes for single samples | Native batch support |
| Interface | `forward(x, mass, L, evals, evecs, gradX, gradY)` | `forward(surface)` (PyG Data) |
| Implementation | Original nmwsharp/diffusion-net | Enhanced pvnieo/diffusion-net-plus |

## Technical Notes

### Why batch_size=1?
- Traditional DiffusionNet doesn't support batching
- batch_size=1 allows fair comparison (both process one sample at a time)
- DiffusionNet++ should produce identical results at batch_size=1

### Wrapper Necessity
The wrapper (`DiffusionNetBlockPyGWrapper`) is necessary because:
1. atomsurf uses PyTorch Geometric Data objects
2. Traditional DiffusionNet expects separate tensor arguments
3. PyG uses sparse tensors, traditional expects dense
4. Mass matrix stored as diagonal sparse matrix → needs extraction

### Validation
If DiffusionNet++ is correctly implemented:
- Both models should converge to similar accuracy
- Loss curves should be comparable
- Final validation metrics should match (within numerical precision)

---

## Quick Start Commands

**Start both trainings (detached):**
```bash
# Traditional
tmux new -s trad_diffnet -d "source /root/miniconda3/etc/profile.d/conda.sh && conda activate atomsurf_traditional && cd /root/atomsurf/atomsurf/tasks/masif_ligand && PYTHONPATH=/root/atomsurf/diffusion-net/src:\$PYTHONPATH python3 train.py encoder=surf_only_traditional.yaml optimizer.lr=0.0001 scheduler=reduce_lr_on_plateau epochs=100 loader.batch_size=1 loader.num_workers=16 diffusion_net.use_bn=false diffusion_net.use_layernorm=true diffusion_net.init_time=2.0 diffusion_net.init_std=2.0 train.save_top_k=5 train.early_stoping_patience=500 min_batch_size=1 run_name=traditional_diffnet_bs1 device=2"

# DiffusionNet++
tmux new -s plus_diffnet -d "source /root/miniconda3/etc/profile.d/conda.sh && conda activate atomsurf && cd /root/atomsurf/atomsurf/tasks/masif_ligand && python3 train.py encoder=surf_only.yaml optimizer.lr=0.0001 scheduler=reduce_lr_on_plateau epochs=100 loader.batch_size=1 loader.num_workers=16 diffusion_net.use_bn=false diffusion_net.use_layernorm=true diffusion_net.init_time=2.0 diffusion_net.init_std=2.0 train.save_top_k=5 train.early_stoping_patience=500 min_batch_size=1 run_name=plus_diffnet_bs1 device=3"

# Check status
tmux ls
```

**Monitor logs:**
```bash
# Traditional
tail -f outputs/traditional_diffnet_bs1/train.log

# DiffusionNet++
tail -f outputs/plus_diffnet_bs1/train.log
```

