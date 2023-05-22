
"""A script to train a model on the train dataset."""

#%%
import os
import util.utility as ul
from loss.loss import MyLoss
from model.custom_model import Model
from config.configClass import Config
import hyperparameter as p
from tester.tester import Tester
from dataset.dataset import DataSet
from trainer.trainer import Trainer
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchmetrics.classification as cf
from tqdm.auto import tqdm
import wandb

load_path = os.path.join(p.SAVED_MODELS_DIR, f"{p.train_config.model_name}.pt")
save_path = load_path

def main(config:Config):
    """Main function of the script."""

    # setup model and other components
    model = Model(config)                                                          # create model
    model.load_state_dict(torch.load(load_path))                                   # load model
    optimizer = AdamW(model.parameters(), lr=config.lr)                            # create optimizer
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.weight_decay)   # create scheduler
    criterion = MyLoss()                                                           # create criterion
    evaluator = cf.BinaryAccuracy()
    evaluator2 = cf.BinaryF1Score()
    if config.WandB:
        wandb.watch(models=model, criterion=criterion, log='all', log_freq=1)

    # prepare dataset and dataloader
    train_set:DataSet = DataSet("train")    # create train dataset
    valid_set:DataSet = DataSet("valid")    # create valid dataset
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)  # create train dataloader
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)  # create valid dataloader

    # create trainer and tester
    trainer = Trainer(model=model, config=config, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    valider = Tester(model=model, config=config, test_loader=valid_loader)
    valider.add_evaluator(evaluator)
    valider.add_evaluator(evaluator2)

    # start training
    train_result = dict(train_loss=0)
    valid_result = dict.fromkeys(valider.evaluators.keys(), 0)
    for epoch in range(0,config.epochs):
        ul.show_result(config, epoch, train_result, valid_result)          # show result
        train_result = trainer.train()                                     # train a epoch
        valid_result = valider.eval()                                      # validate a epoch
        scheduler.step()                                                   # update learning rate
        if config.WandB:
            ul.log_result(epoch, train_result, valid_result)               # log result to wandb
    ul.show_result(config, config.epochs, train_result, valid_result)      # show result

    # store model
    if config.SAVE:
        ul.store_model(config, model, save_path)

if __name__ == '__main__':
    #%% print information
    print(f'Torch version: {torch.__version__}')
    ul.init(p.train_config)
    DataSet.load_data(p.train_config)

    #%% start training
    main(p.train_config)
    ul.show_time()

    #%% close dataset
    DataSet.close()