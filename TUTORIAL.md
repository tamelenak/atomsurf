Tutorial to set up conda environment and run the masif site task.

Installation

```bash
conda conda env create -f environment.yml
conda activate atomsurf
pip install git+https://github.com/pvnieo/diffusion-net-plus.git
```
For notebook support
```bash
python -m ipykernel install --user --name atomsurf --display-name "Python (atomsurf)"
```
Other packages
```bash
# for some reason some packages are not installed. Do this:
pip install fair-esm biopython omegaconf deltaconv potpourri3d hydra open3d
conda install cudatoolkit=11.7 -c nvidia
conda install pytorch=1.13 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.3.0 pytorch-scatter pytorch-sparse pytorch-spline-conv pytorch-cluster -c pyg
pip install pyg-lib==0.3.1 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install pytorch-lightning==1.8.6
python -c "import torch; print(torch.cuda.is_available())"
```
Test if environment is correctly set up by running a minimal script
```bash
python example.py
apt-get update && apt-get install -y pdb2pqr
```

Masif Site Task

- Download masif site pdbs and surfaces
- Put them in data/masif_site
- Folders need to be called 01-benchmark_pdbs and 01-benchmark_surfaces OR names have to be changed in the script
- Set num_workers in masif_site_test.py to number of available CPU workers - 2

Run script
```bash
python masif_site_test.py --batch_size 1024 --epochs 10
```