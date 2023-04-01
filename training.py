
"""A script to train a model on the train dataset."""

import config.global_variables as gv
import util.utility as ul
from model.model import WaveFuncTrans
from trainer.trainer import Trainer
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def main():
    """Main function of the script."""
    # set random seed
    seed:int = 0
    ul.set_seed(seed=seed)

    # create model and variables
    model:nn.Module = WaveFuncTrans()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs:int = 10
    batch_size:int = 32
    learning_rate:float = 0.001
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    warmup_ratio:float = 0.95
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=warmup_ratio)
    criterion:nn.Module = nn.MSELoss()
    score:nn.Module = nn.MSELoss()

    # create dataloader
    train_set:Dataset = Dataset()
    valid_set:Dataset = Dataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # create trainer
    trainer:Trainer = Trainer(model,
                              device,
                              train_loader,
                              valid_loader,
                              epochs,
                              scheduler,
                              optimizer,
                              criterion,
                              score)

    # start training
    print('Training model...')
    train_result = trainer.train()
    print('Finished training model.')

    # save model
    torch.save(model.state_dict(), gv.SAVED_MODELS_DIR + f'/model_loss_{train_result[0][-1]:.3f}_score_{train_result[2][-1]:.3f}.pt')
    
if __name__ == '__main__':
    main()
