
"""A script to train a model on the train dataset."""

#%%
import os
import util.utility as ul
import global_var.path as p
from loss.loss import MyLoss
from model.model import Model
from config.config import Config
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

#%%
# create config
config = Config(
    project_name = 'NN_Template',
    model_name = 'NN_test',
    seed = 0,
    input_size = 8,
    output_size = 10,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size = 16,
    epochs = 100,
    lr = 0.0001,
    weight_decay = 0.999,
    split_ratio = 0.2,
    SAVE = False,
    WandB = False
)

def init(config:Config):
    """Initialize the script."""
    # create directory
    os.makedirs(p.SAVED_MODELS_DIR, exist_ok=True)
    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    # set random seed
    ul.set_seed(seed=config.seed)
    # initialize wandb
    if config.WandB:
        wandb.init(project=config.project_name, name=config.model_name, config=config.data)

def show_result(config:Config, epoch:int, train_result:dict, valid_result:dict):
    """Print result of training and validation."""
    # print result
    os.system('clear')
    print(f'Epoch: ({epoch} / {config.epochs})')
    print("Train result:")
    print(f'\ttrain_loss: {train_result["train_loss"]}')
    print("Valid result:")
    for name,score in valid_result.items():
        print(f'\t{name}: {score}')

def store_result(config:Config, epoch:int, train_result:dict, valid_result:dict):
    """Store the result."""
    if config.WandB:
        wandb.log(train_result, step=epoch)
        wandb.log(valid_result, step=epoch)

def store_model(config:Config, model:nn.Module):
    """Store the model."""
    # save model
    SAVE_MODEL_PATH = p.SAVED_MODELS_DIR + f"/{config.model_name}.pt"
    print(f'Saving model to {SAVE_MODEL_PATH}')
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    if config.WandB:
        wandb.save(SAVE_MODEL_PATH)

def main(config:Config):
    """Main function of the script."""

    # setup model and other components
    model = Model(config)                                                         # create model
    #model.load_state_dict(torch.load(p.SAVED_MODELS_DIR + '/QINN_test.pt'))       # load model
    optimizer = AdamW(model.parameters(), lr=config.lr)                           # create optimizer
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.weight_decay)  # create scheduler
    criterion = nn.CrossEntropyLoss()                                             # create criterion
    evaluator = cf.MulticlassAccuracy(num_classes=config.output_size,average='micro', validate_args=True) # create evaluator1
    evaluator2 = cf.MulticlassF1Score(num_classes=config.output_size,average='micro', validate_args=True) # create evaluator2
    if config.WandB:
        wandb.watch(models=model, criterion=criterion, log='all', log_freq=500)

    # prepare dataset and dataloader
    DataSet.load_data(config)               # initialize dataset
    train_set:DataSet = DataSet("train")    # create train dataset
    valid_set:DataSet = DataSet("valid")    # create valid dataset
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)  # create train dataloader
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, pin_memory=True) # create valid dataloader

    # create trainer and tester
    trainer = Trainer(model=model, config=config, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    valider = Tester(model=model, config=config, test_loader=valid_loader)
    valider.add_evaluator(evaluator)

    # start training
    train_result = dict(train_loss=0)
    valid_result = dict.fromkeys(valider.evaluators.keys(), 0)
    for epoch in tqdm(range(1,config.epochs+1), desc="Epoch"):
        show_result(config, epoch, train_result, valid_result)  # show result
        train_result = trainer.train()                          # train a epoch
        valid_result = valider.eval()                           # validate a epoch
        scheduler.step()                                        # update learning rate
        store_result(config, epoch, train_result, valid_result) # store result

    # store model
    if config.SAVE:
        store_model(config, model)


if __name__ == '__main__':
    # print information
    print(f'Torch version: {torch.__version__}')

    # initialize
    init(config)

    # start training
    main(config)

