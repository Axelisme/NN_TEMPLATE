# NEURAL NETWORK TEMPLATE

## Usage
### generate dataset
```bash
python scripts/{xxx}_generator.py
```
### start new train
Default model config path: `configs/modelrc.yaml`  
Default task config path: `configs/taskrc.yaml`
```bash
python scripts/run_model.py -m train
                            -n {name}
                            [-M {modelrc}]
                            [-T {taskrc}]
                            [-o {ckpt_dir}]
                            [--WandB]
                            [--slient]
                            [--overwrite]
```
### resume train
```bash
python scripts/run_model.py -m train -e {ckpt} -a
```
### evaluate
```bash
python scripts/run_model.py -m evaluate -e {ckpt} -t {loader1} {loader2} ...
```
### sweep
Default model config path: `configs/modelrc.yaml`  
Default task config path: `configs/taskrc.yaml`
```bash
python scripts/sweep.py -s {sweeprc}
                        [-M {modelrc}]
                        [-T {taskrc}]
```
