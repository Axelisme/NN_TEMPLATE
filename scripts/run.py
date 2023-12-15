import os
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

from global_vars import RESULT_DIR
from util.io import show, show_train_result, show_valid_result
from util.utility import init, check_better, create_instance
from modules.trainer import Trainer
from modules.valider import Valider
from modules.ckptmanager import CheckPointManager
from custom.evaluator.Loss2evaluator import LossScore


def get_model(model_conf:Dict[str, Dict|Any]):
    model_select = model_conf['select']
    show(f"[INFO] Using {model_select} as model.")
    return create_instance(model_conf[model_select])


def get_modules_for_train(arch_conf:Dict[str, Dict|Any], parameters):
    # select optimizer
    optim_select = arch_conf['optimizer']['select']
    show(f"[INFO] Using {optim_select} as optimizer.")
    optimizer = create_instance(arch_conf['optimizer'][optim_select], params=parameters)

    # select scheduler
    sched_select = arch_conf['scheduler']['select']
    show(f"[INFO] Using {sched_select} as scheduler.")
    scheduler = create_instance(arch_conf['scheduler'][sched_select], optimizer=optimizer)

    # select loss function
    loss_select = arch_conf['loss']['select']
    show(f"[INFO] Using {loss_select} as loss function.")
    criterion = create_instance(arch_conf['loss'][loss_select])

    return optimizer, scheduler, criterion


def get_modules_for_eval(metric_conf:Dict[str, Dict|Any], criterion=None):
    # select evaluator
    metrics = {}
    for name in metric_conf['select']:
        show(f"[INFO] Using {name} as metric.")
        # use importlib to avoid weird bug of 'BinnedAveragePrecision' not found
        metrics[name] = create_instance(metric_conf[name])
    if criterion is not None:
        show(f"[INFO] Using LossScore as metric.")
        metrics['valid_loss'] = LossScore(criterion)

    return MetricCollection(metrics)


def get_dataloader(datasets_conf:Dict, loader_conf:Dict, mode:str):
    dataset = loader_conf['dataset']
    show(f"[INFO] Using {dataset} as {mode} dataset.")
    dataset = create_instance(datasets_conf[dataset])
    return DataLoader(dataset, **loader_conf['kwargs'])


def start_train(args: Namespace, conf: Dict[str, Dict|Any]):
    """Training model base on given config."""
    # load models
    arch_conf = conf['architectures']
    model = get_model(arch_conf['model'])
    optimizer, scheduler, criterion = get_modules_for_train(arch_conf, model.parameters())
    if arch_conf['metric']['use_loss']:
        metrics = get_modules_for_eval(arch_conf['metric'], criterion)
    else:
        metrics = get_modules_for_eval(arch_conf['metric'])

    # load model and optimizer from checkpoint if needed
    if args.load or not args.disable_save:
        ckpt_conf = conf['ckpt']
        if ckpt_conf['ckpt_dir'] is None:
            ckpt_conf['ckpt_dir'] = os.path.join(RESULT_DIR, args.name)
        ckpt_manager = CheckPointManager(**ckpt_conf)
        show(f"[INFO] Checkpoint directory: {ckpt_manager.ckpt_dir}")
        save_conf = {'args': vars(args), 'conf': conf}
        ckpt_manager.save_config(
            save_conf = save_conf,
            config_name=f"train_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        if args.load:
            show(f"[INFO] Loading checkpoint from {args.load}")
            ckpt_manager.load_model(model, ckpt_path=args.load)

    # register model to wandb if needed
    if args.WandB and not args.not_track_params:
        wandb.watch(models=model, criterion=criterion, **conf['WandB']['watch_kwargs'])

    # prepare dataset and dataloader
    datasets_conf = conf['data']['datasets']
    loader_conf = conf['data']['dataloaders']
    train_loader = get_dataloader(datasets_conf, loader_conf[args.train_loader], 'train')
    valid_loader = get_dataloader(datasets_conf, loader_conf[args.valid_loader], 'valid')

    # create trainer and valider
    runner_conf = conf['runner']
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        args=args,
        **runner_conf['trainer_kwargs']
    )
    valider = Valider(
        model=model,
        valid_loader=valid_loader,
        metrics=metrics,
        args=args,
        **runner_conf['valider_kwargs']
    )

    # start training
    best_result = {}
    pbar = tqdm(total=runner_conf['total_steps'], desc='Overall', dynamic_ncols=True, disable=args.slient)
    for step in range(1, runner_conf['total_steps'] + 1):
        if runner_conf['step_mod'] == 'grad':
            trainer.one_step()
        elif runner_conf['step_mod'] == 'epoch':
            trainer.one_epoch()

        pbar.update()

        if step % runner_conf['log_freq'] == 0 or step == runner_conf['total_steps']:
            train_result = trainer.pop_result()
            show_train_result(runner_conf, step, trainer.lr, train_result)
            if args.WandB:
                wandb.log(train_result, step=step, commit=False)

        if step % runner_conf['dev_freq'] == 0 or step == runner_conf['total_steps']:
            valider.one_epoch()
            current_result = valider.pop_result()
            show_valid_result(runner_conf, step, current_result)
            if args.WandB:
                wandb.log(current_result, step=step, commit=False)
            if not args.disable_save:
                if check_better(ckpt_conf, current_result, best_result):
                    best_result = current_result
                    ckpt_manager.save_model(
                        model=model,
                        ckpt_name='dev-best.pth',
                        meta_conf={'step': step, 'conf': save_conf, 'result': best_result},
                        overwrite=True
                    )

        if step % runner_conf['save_freq'] == 0 and not args.disable_save:
            ckpt_manager.save_model(
                model=model,
                ckpt_name=f'dev-step-{step}.pth',
                meta_conf={'step': step, 'conf': save_conf, 'result': None},
                overwrite=args.overwrite
            )

        if args.WandB:
            wandb.log({'lr':trainer.lr}, step=step, commit=True)
    pbar.close()

    trainer.close()
    valider.close()


def start_evaluate(args: Namespace, conf: Dict[str, Dict|Any]):
    """Valid model base on given config."""
    # load models
    arch_conf = conf['architectures']
    model = get_model(arch_conf['model'])
    metrics = get_modules_for_eval(arch_conf['metric'])

    # load model and optimizer from checkpoint if needed
    ckpt_conf = conf['ckpt']
    if ckpt_conf['ckpt_dir'] is None:
        ckpt_conf['ckpt_dir'] = os.path.join(RESULT_DIR, args.name)
    ckpt_manager = CheckPointManager(**ckpt_conf)
    show(f"[INFO] Checkpoint directory: {ckpt_manager.ckpt_dir}")
    ckpt_manager.load_model(model, ckpt_path=args.load)

    # prepare dataset and dataloader
    loader_conf = conf['data']['dataloaders']
    valid_loader = get_dataloader(conf['data']['datasets'], loader_conf[args.valid_loader], 'valid')

    # create valider
    runner_conf = conf['runner']
    valider = Valider(
        model=model,
        valid_loader=valid_loader,
        metrics=metrics,
        args=args,
        **runner_conf['valider_kwargs']
    )

    # start validating
    valider.one_epoch()
    current_result = valider.pop_result()
    show_valid_result(runner_conf, 0, current_result)

    valider.close()


def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-n', '--name', type=str, required=True, help='name of this training')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'evaluate'], required=True, help='mode of this training')
    parser.add_argument('-c', '--config', type=str, default='configs/template.yaml', help='path to config file')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('--train_loader', type=str, default='train_loader', help='train loader name')
    parser.add_argument('--valid_loader', type=str, default='valid_loader', help='valid loader name')
    parser.add_argument('--load', default=None, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--slient', action='store_true', help='slient or not')

    parser.add_argument('--WandB', action='store_true', help='use W&B to log')
    parser.add_argument('--not_track_params', action='store_true', help='not track params in W&B')
    parser.add_argument('--disable_save', action='store_true', help='disable auto save ckpt')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing ckpt or not')
    args = parser.parse_args()

    show.set_slient(args.slient)

    # load config
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    # initialize
    init(args.seed)

    # start training or validating
    if args.mode == 'train':
        if args.WandB:
            show(f'[INFO] Using W&B to log.')
            wandb.init(
                project=conf['WandB']['project'],
                name=args.name,
                **conf['WandB']['init_kwargs']
            )

        start_train(args, conf)
    elif args.mode == 'evaluate':
        assert args.load is not None, "load should not be None when mode is evaluate."
        assert not args.disable_save, "disable_save should be False when mode is evaluate."
        assert not args.WandB, "WandB should be False when mode is evaluate."
        assert not args.not_track_params, "not_track_params should be False when mode is evaluate."
        assert not args.overwrite, "overwrite should be False when mode is evaluate."

        start_evaluate(args, conf)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    main()
