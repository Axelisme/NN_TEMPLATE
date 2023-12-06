# NEURAL NETWORK TEMPLATE

Dependence: `torch`, `tqdm`, `torchmetric`, `wandb`, `numpy`,`pyyaml`

## Usage
### generate hdf5 dataset
```bash
python -m scripts.hdf5_generator
```
### train
Default config path: `configs/template.yaml`
```bash
python -m scripts.training -n {name}
                           [-c {config_path}]
                           [-s {seed}]
                           [--WandB]
                           [--load {path}]
                           [--slient]
                           [--overwrite]
```
### sweep
```bash
python -m scripts.sweep -c {config_path}
```
