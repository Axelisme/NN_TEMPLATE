
"""A script to train a model on the train dataset."""

#%%
import os
import wandb
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchmetrics.classification as cf
from util.utility import init
from util.tool import measure_time, show_time
from util.io import logit
from util.checkpoint import load_checkpoint, save_checkpoint
from hyperparameters import *
from model.customModel import CustomModel
from tester.tester import Tester
from dataset.customDataset import CustomDataSet
from trainer.trainer import Trainer
from config.configClass import Config


#@logit(LOG_CONSOLE)
@measure_time
def start_train(conf:Config):
    """Main function of the script."""

    # setup model and other components
    model = CustomModel(conf).to(torch.device(conf.device))                                 # create model
    optimizer = AdamW(model.parameters(), lr=conf.lr)                                       # create optimizer
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=conf.gamma)                     # create scheduler
    criterion = nn.CrossEntropyLoss()                                                       # create criterion
    evaluator = cf.MulticlassAccuracy(num_classes=conf.output_size, average='macro')        # create evaluator1
    load_checkpoint(model,
                    optimizer,
                    scheduler,
                    checkpoint_path=conf.load_path,
                    device=torch.device(conf.device))     # load model and optimizer

    # register model to wandb
    if conf.WandB and not hasattr(conf,"Sweep"):
        wandb.watch(models=model, criterion=criterion, log="gradients", log_freq=100)

    # prepare dataset and dataloader
    dataset_name = "dataset_all.hdf5"
    train_set = CustomDataSet(conf, "train", dataset_name)    # create train dataset
    valid_set = CustomDataSet(conf, "valid", dataset_name)    # create valid dataset
    train_loader = DataLoader(dataset=train_set,
                              batch_size=conf.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=conf.num_workers)  # create train dataloader
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=conf.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=conf.num_workers)  # create valid dataloader

    # create trainer and tester
    trainer = Trainer(model=model, config=conf, loader=train_loader, optimizer=optimizer, criterion=criterion)
    valider = Tester(model=model, config=conf, loader=valid_loader, criterion=criterion)
    valider.add_evaluator(evaluator, name="accuracy")

    # start training
    train_result = None
    valid_result = None
    best_score = next(iter(valider.eval().values())).item() if conf.save_path else 0
    for epoch in range(1,conf.epochs+1):
        print('-'*79)
        train_result = trainer.train()                                                  # train a epoch
        valid_result = valider.eval()                                                   # validate a epoch
        lr = optimizer.param_groups[0]["lr"]
        show_result(conf, epoch, train_result, valid_result, lr)                        # show result
        if hasattr(conf,"WandB") and conf.WandB:                                        # log result to wandb
            wandb.log({'lr':lr}, step=epoch, commit=False)
            wandb.log({'train_loss':train_result}, step=epoch, commit=False)
            wandb.log(valid_result, step=epoch, commit=True)
        cur_score = next(iter(valid_result.values())).item()
        scheduler.step()                                                                # update learning rate
        if cur_score > best_score:                                                      # save best model
            save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path=conf.save_path, overwrite=True)
            best_score = cur_score


def show_result(conf:Config, epoch, train_result, valid_result:dict, lr) -> None:
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
    init(train_conf.seed)
    if hasattr(train_conf,"WandB") and train_conf.WandB:
        wandb.init(project=train_conf.project_name,
                   name=train_conf.model_name,
                   config=train_conf.data)

    #%% start training
    start_train(train_conf)
    show_time()
