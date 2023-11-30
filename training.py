import time
import yaml
import wandb
import argparse
import importlib
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from util.utility import init
from trainer.trainer import Trainer
from valider.valider import Valider
from evaluator.Loss2evaluator import LossScore
from ckptmanager.manager import CheckPointManager

def get_models(arch_conf):
    # select model
    model_select = arch_conf['model']['select']
    print(f"[INFO] Using {model_select} as model.")
    model_module = importlib.import_module(arch_conf['model'][model_select]['module'])
    model = getattr(model_module, model_select)(**arch_conf['model'][model_select]['args'])

    # select optimizer
    optim_select = arch_conf['optimizer']['select']
    print(f"[INFO] Using {optim_select} as optimizer.")
    optim_module = importlib.import_module(arch_conf['optimizer'][optim_select]['module'])
    optimizer = getattr(optim_module, optim_select)(
        model.parameters(), **arch_conf['optimizer'][optim_select]['args'])

    # select scheduler
    sched_select = arch_conf['scheduler']['select']
    print(f"[INFO] Using {sched_select} as scheduler.")
    sched_module = importlib.import_module(arch_conf['scheduler'][sched_select]['module'])
    scheduler = getattr(sched_module, sched_select)(
        optimizer, **arch_conf['scheduler'][sched_select]['args'])

    # select loss function
    loss_select = arch_conf['loss']['select']
    print(f"[INFO] Using {loss_select} as loss function.")
    criter_module = importlib.import_module(arch_conf['loss'][loss_select]['module'])
    criterion = getattr(criter_module, loss_select)(**arch_conf['loss'][loss_select]['args'])

    # select evaluator
    metric_selects = arch_conf['metric']['select']
    metrics = {}
    for name in metric_selects:
        print(f"[INFO] Using {name} as metric.")
        # use importlib to avoid weird bug of 'BinnedAveragePrecision' not found
        metric_module = importlib.import_module(arch_conf['metric'][name]['module'])
        metrics[name] = getattr(metric_module, name)(**arch_conf['metric'][name]['args'])
    if arch_conf['metric']['use_loss']:
        print(f"[INFO] Using loss as metric.")
        metrics['Loss'] = LossScore(criterion)
    metrics = MetricCollection(metrics)

    return model, optimizer, scheduler, criterion, metrics


def get_dataloader(data_conf):
    def get_dataset(dataset_conf, mode):
        dataset_select = dataset_conf['select']
        print(f"[INFO] Using {dataset_select} as {mode} dataset.")
        dataset_module = importlib.import_module(dataset_conf[dataset_select]['module'])
        return getattr(dataset_module, dataset_select)(**dataset_conf[dataset_select]['args'])

    train_set = get_dataset(data_conf['train_loader']['dataset'], 'train')
    valid_set = get_dataset(data_conf['valid_loader']['dataset'], 'valid')
    train_loader = DataLoader(dataset=train_set, **data_conf['train_loader']['args'])
    valid_loader = DataLoader(dataset=valid_set, **data_conf['valid_loader']['args'])

    return train_loader, valid_loader


def start_train(args, conf):
    """Training model base on given config."""
    # load models
    arch_conf = conf['architectures']
    model, optimizer, scheduler, criterion, metrics = get_models(arch_conf)

    # load model and optimizer from checkpoint if needed
    ckpt_conf = conf['ckpt']
    save_conf = {'args':vars(args), 'config':conf}
    ckpt_manager = CheckPointManager(save_conf, args.name, model, **ckpt_conf)
    print(f"[INFO] Saving checkpoint to {ckpt_manager.ckpt_dir}")
    ckpt_manager.save_config(f"train_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
    if args.load is not None:
        print(f"[INFO] Loading checkpoint from {args.load}")
        ckpt_manager.load(ckpt_path=args.load)

    # register model to wandb if needed
    if args.WandB:
        wandb.watch(models=model, criterion=criterion, **conf['WandB']['watch_args'])

    # prepare dataset and dataloader
    data_conf = conf['data']
    train_loader, valid_loader = get_dataloader(data_conf)

    # create trainer and valider
    runner_conf = conf['runner']
    trainer = Trainer(model, train_loader, optimizer, criterion, args, **runner_conf['trainer'])
    valider = Valider(model, valid_loader, metrics, args, **runner_conf['valider'])

    # start training
    for epoch in range(1, runner_conf['epochs']+1):
        print('-'*79)

        train_result = trainer.fit()
        valid_result = valider.eval()

        lr = optimizer.param_groups[0]['lr']
        show_result(runner_conf, epoch, lr, train_result, valid_result)

        scheduler.step()

        ckpt_manager.update(valid_result, epoch, args.overwrite)

        if args.WandB:
            wandb.log({'lr':lr}, step=epoch, commit=False)
            wandb.log(train_result, step=epoch, commit=False)
            wandb.log(valid_result, step=epoch, commit=True)


def show_result(conf:dict, epoch, lr, train_result:dict, valid_result:dict):
    """Print result of training and validation."""
    # print result
    print(f'Epoch: ({epoch} / {conf["epochs"]})')
    print(f'lr: {lr:0.3e}')
    print("Train result:")
    for name, evaluator in train_result.items():
        print(f'\t{name}: {evaluator:0.4f}')
    print("Valid result:")
    for name, evaluator in valid_result.items():
        print(f'\t{name}: {evaluator:0.4f}')


def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-n', '--name', type=str, required=True, help='name of this training')
    parser.add_argument('-c', '--config', type=str, default='config/template.yaml', help='path to config file')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('--WandB', action='store_true', help='use W&B to log')
    parser.add_argument('--load', default=None, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing ckpt or not')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    # initialize
    init(args.seed)
    if args.WandB:
        print(f'[INFO] Using W&B to log.')
        wandb.init(**conf['WandB']['init_args'])

    # start training
    start_train(args, conf)


if __name__ == '__main__':
    main()
