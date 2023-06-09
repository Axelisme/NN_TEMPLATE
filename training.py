
"""A script to train a model on the train dataset."""

#%%
from typing import Dict
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchmetrics.classification as cf
from util.utility import init
from hyperparameters import train_conf
from model.customModel import CustomModel
from evaluator.Loss2evaluator import LossScore
from tester.tester import Tester
from dataset.customDataset import CustomDataSet
from trainer.trainer import Trainer
from config.configClass import Config
from ckptmanager.manager import CheckPointManager


def start_train(conf:Config):
    """Training model base on given config."""

    # device setting
    device = torch.device(conf.device)

    # setup model and other components
    model = CustomModel(conf)                                                               # create model
    optimizer = AdamW(model.parameters(), lr=conf.init_lr)                                  # create optimizer
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=conf.gamma)                     # create scheduler
    criterion = nn.CrossEntropyLoss()                                                       # create criterion
    evaluator = cf.MulticlassAccuracy(num_classes=conf.output_size, average='macro')        # create evaluator

    # load model and optimizer from checkpoint if needed
    ckpt_manager = CheckPointManager(conf, model, optim=optimizer, scheduler=scheduler)
    if conf.Load:
        ckpt_manager.load(ckpt_path=conf.load_path, device=device)

    # register model to wandb if needed
    if conf.WandB and not hasattr(conf,"Sweep"):
        wandb.watch(models=model, criterion=criterion, log="gradients", log_freq=100)

    # prepare dataset and dataloader
    dataset_name = conf.dataset_name
    train_set = CustomDataSet(conf, "train", dataset_name)    # create train dataset
    valid_set = CustomDataSet(conf, "valid", dataset_name)    # create valid dataset
    batch_size = conf.batch_size
    num_workers = conf.num_workers
    train_loader = DataLoader(dataset     = train_set,
                              batch_size  = batch_size,
                              shuffle     = False,
                              pin_memory  = True,
                              num_workers = num_workers)  # create train dataloader
    valid_loader = DataLoader(dataset     = valid_set,
                              batch_size  = batch_size,
                              shuffle     = False,
                              pin_memory  = True,
                              num_workers = num_workers)  # create valid dataloader

    # create trainer and valider
    trainer = Trainer(model, device, train_loader, optimizer, criterion)
    valider = Tester(model, device, valid_loader)
    valider.add_evaluator(evaluator, name="accuracy")
    valider.add_evaluator(LossScore(criterion), name="val_loss")


    # start training
    for epoch in range(1,conf.epochs+1):
        print('-'*79)

        train_result = trainer.fit()                                                # train a epoch
        valid_result = valider.eval()                                               # validate a epoch

        lr = scheduler.get_last_lr()[-1]                                            # get current learning rate
        show_result(conf, epoch, lr, train_result, valid_result)                    # show result

        scheduler.step()                                                            # update learning rate

        if conf.Save:                                                               # save checkpoint if needed
            cur_score = get_one(valid_result).item()                                # get current score
            ckpt_manager.update(cur_score, epoch)                                   # save checkpoint if better

        if hasattr(conf,"WandB") and conf.WandB:                                    # log result to wandb
            wandb.log({'lr':lr}, step=epoch, commit=False)
            wandb.log({'train_loss':train_result}, step=epoch, commit=False)
            wandb.log(valid_result, step=epoch, commit=True)


def get_one(dict:Dict):
    """Get the first element of a dict."""
    return next(iter(dict.values()))


def show_result(conf:Config, epoch, lr, train_result, valid_result:dict) -> None:
    """Print result of training and validation."""
    # print result
    print(f'Epoch: ({epoch} / {conf.epochs})')
    print(f'lr: {lr:0.3e}')
    print("Train result:")
    print(f'\ttrain_loss: {train_result:0.4f}')
    print("Valid result:")
    for name, evaluator in valid_result.items():
        print(f'\t{name}: {evaluator:0.4f}')


if __name__ == '__main__':
    #%% print information
    print(f'Torch version: {torch.__version__}')
    # initialize
    init(train_conf.seed)
    if hasattr(train_conf,"WandB") and train_conf.WandB:
        wandb.init(project=train_conf.project_name,
                   name=train_conf.model_name,
                   config=train_conf.as_dict())

    #%% start training
    start_train(train_conf)
