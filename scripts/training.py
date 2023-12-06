import time
import yaml
import wandb
import argparse
import importlib
from typing import Dict, Any
from argparse import Namespace
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm.auto import tqdm

from util.io import show
from util.utility import init
from modules.runner import Runner
from modules.ckptmanager import CheckPointManager
from custom.evaluator.Loss2evaluator import LossScore

def get_models(arch_conf:Dict[str, Dict|Any]):
    # select model
    model_select = arch_conf['model']['select']
    show(f"[INFO] Using {model_select} as model.")
    model_module = importlib.import_module(arch_conf['model'][model_select]['module'])
    model = getattr(model_module, model_select)(**arch_conf['model'][model_select]['args'])

    # select optimizer
    optim_select = arch_conf['optimizer']['select']
    show(f"[INFO] Using {optim_select} as optimizer.")
    optim_module = importlib.import_module(arch_conf['optimizer'][optim_select]['module'])
    optimizer = getattr(optim_module, optim_select)(
        model.parameters(), **arch_conf['optimizer'][optim_select]['args'])

    # select scheduler
    sched_select = arch_conf['scheduler']['select']
    show(f"[INFO] Using {sched_select} as scheduler.")
    sched_module = importlib.import_module(arch_conf['scheduler'][sched_select]['module'])
    scheduler = getattr(sched_module, sched_select)(
        optimizer, **arch_conf['scheduler'][sched_select]['args'])

    # select loss function
    loss_select = arch_conf['loss']['select']
    show(f"[INFO] Using {loss_select} as loss function.")
    criter_module = importlib.import_module(arch_conf['loss'][loss_select]['module'])
    criterion = getattr(criter_module, loss_select)(**arch_conf['loss'][loss_select]['args'])

    # select evaluator
    metric_selects = arch_conf['metric']['select']
    metrics = {}
    for name in metric_selects:
        show(f"[INFO] Using {name} as metric.")
        # use importlib to avoid weird bug of 'BinnedAveragePrecision' not found
        metric_module = importlib.import_module(arch_conf['metric'][name]['module'])
        metrics[name] = getattr(metric_module, name)(**arch_conf['metric'][name]['args'])
    if arch_conf['metric']['use_loss']:
        show(f"[INFO] Using loss as metric.")
        metrics['valid_loss'] = LossScore(criterion)
    metrics = MetricCollection(metrics)

    return model, optimizer, scheduler, criterion, metrics


def get_dataloader(data_conf:Dict[str, Dict|Any]):
    def get_dataset(dataset_conf, mode):
        dataset_select = dataset_conf['select']
        show(f"[INFO] Using {dataset_select} as {mode} dataset.")
        dataset_module = importlib.import_module(dataset_conf[dataset_select]['module'])
        return getattr(dataset_module, dataset_select)(**dataset_conf[dataset_select]['args'])

    train_set = get_dataset(data_conf['train_loader']['dataset'], 'train')
    valid_set = get_dataset(data_conf['valid_loader']['dataset'], 'valid')
    train_loader = DataLoader(dataset=train_set, **data_conf['train_loader']['args'])
    valid_loader = DataLoader(dataset=valid_set, **data_conf['valid_loader']['args'])

    return train_loader, valid_loader


def start_train(args: Namespace, conf: Dict[str, Dict|Any]):
    """Training model base on given config."""
    # load models
    arch_conf = conf['architectures']
    model, optimizer, scheduler, criterion, metrics = get_models(arch_conf)

    # load model and optimizer from checkpoint if needed
    if args.load or not args.disable_save:
        ckpt_conf = conf['ckpt']
        save_conf = {'args':vars(args), 'config':conf}
        ckpt_manager = CheckPointManager(save_conf, args.name, model, **ckpt_conf)
        show(f"[INFO] Saving checkpoint to {ckpt_manager.ckpt_dir}")
        ckpt_manager.save_config(f"train_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
        if args.load:
            show(f"[INFO] Loading checkpoint from {args.load}")
            ckpt_manager.load(ckpt_path=args.load)

    # register model to wandb if needed
    if args.WandB and not args.not_track_params:
        wandb.watch(models=model, criterion=criterion, **conf['WandB']['watch_args'])

    # prepare dataset and dataloader
    data_conf = conf['data']
    train_loader, valid_loader = get_dataloader(data_conf)

    # create trainer and valider
    runner_conf = conf['runner']
    runner = Runner(model,
                    train_loader,
                    valid_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    metrics,
                    args,
                    **runner_conf['init_args']
                )

    # start training
    best_result = {}
    pbar = tqdm(total=runner_conf['total_steps'], desc='Overall', dynamic_ncols=True, disable=args.slient)
    for step in range(1, runner_conf['total_steps'] + 1):
        runner.train_one_step()

        pbar.update()

        if step % runner_conf['log_freq'] == 0 or step == runner_conf['total_steps']:
            train_result = runner.pop_train_result()
            show_train_result(runner_conf, step, runner.lr, train_result)
            if args.WandB:
                wandb.log(train_result, step=step, commit=False)

        if step % runner_conf['dev_freq'] == 0 or step == runner_conf['total_steps']:
            runner.valid_one_epoch()
            current_result = runner.pop_valid_result()
            show_valid_result(runner_conf, step, current_result)
            if args.WandB:
                wandb.log(current_result, step=step, commit=False)
            if not args.disable_save:
                if check_better(ckpt_conf, current_result, best_result):
                    best_result = current_result
                    ckpt_manager.save(ckpt_name='dev-best.pth', overwrite=True)

        if step % runner_conf['save_freq'] == 0 and not args.disable_save:
            ckpt_manager.save(ckpt_name=f'dev-step-{step}.pth', overwrite=args.overwrite)

        if args.WandB:
            wandb.log({'lr':runner.lr}, step=step, commit=True)
    pbar.close()


def check_better(conf, current_result, best_result):
    for name in conf['check_metrics']:
        if current_metric := current_result.get(name):
            if best_metric := best_result.get(name):
                if current_metric == best_metric:
                    continue
                return conf['save_mod'] == 'max' and current_metric > best_metric or \
                        conf['save_mod'] == 'min' and current_metric < best_metric
            else:
                return True
        else:
            continue
    return False


def show_train_result(
        conf         : Dict[str,Dict|Any],
        step         : int,
        lr           : float,
        train_result : Dict[str,float]
    ):
    """Print result of training."""
    # print result
    show(f'Step: ({step} / {conf["total_steps"]})')
    show(f'lr: {lr:0.3e}')
    show("Train result:")
    for name, evaluator in train_result.items():
        show(f'\t{name}: {evaluator:0.4f}')

def show_valid_result(
        conf         : Dict[str,Dict|Any],
        step         : int,
        valid_result : Dict[str,float]
    ):
    """Print result of validation."""
    # print result
    show(f'Step: ({step} / {conf["total_steps"]})')
    show("Valid result:")
    for name, evaluator in valid_result.items():
        show(f'\t{name}: {evaluator:0.4f}')


def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-n', '--name', type=str, required=True, help='name of this training')
    parser.add_argument('-c', '--config', type=str, default='configs/template.yaml', help='path to config file')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('--WandB', action='store_true', help='use W&B to log')
    parser.add_argument('--not_track_params', action='store_true', help='not track params in W&B')
    parser.add_argument('--disable_save', action='store_true', help='disable auto save ckpt')
    parser.add_argument('--load', default=None, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--slient', action='store_true', help='slient or not')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing ckpt or not')
    args = parser.parse_args()

    show.set_slient(args.slient)

    # load config
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    # initialize
    init(args.seed)
    if args.WandB:
        show(f'[INFO] Using W&B to log.')
        wandb.init(project=conf['WandB']['project'], **conf['WandB']['init_args'])

    # start training
    start_train(args, conf)


if __name__ == '__main__':
    main()
