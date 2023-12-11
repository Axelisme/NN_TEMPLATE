# NEURAL NETWORK TEMPLATE

Dependence: `torch`, `tqdm`, `torchmetric`, `wandb`, `numpy`,`pyyaml`

## Usage
### generate dataset
```bash
python -m scripts.{xxx}_generator
```
### train
Default config path: `configs/template.yaml`
```bash
python -m scripts.run      -m {train/evaluate}
                           -n {name}
                           [-c {config_path}]
                           [-s {seed}]
                           [--train_loader {loader}]
                           [--valid_loader {loader}]
                           [--WandB]
                           [--load {path}]
                           [--slient]
                           [--overwrite]
```
### sweep
```bash
python -m scripts.sweep -c {config_path}
```
