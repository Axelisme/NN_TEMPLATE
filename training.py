
"""A script to train a model on the train dataset."""

import config.path as p
import util.utility as ul
from model.model import Model
from dataset.dataset import Dataset
from trainer.trainer import Trainer
from evaluator.fl_score import F1Score
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

def main():
    """Main function of the script."""
    # set random seed
    seed:int = 0
    ul.set_seed(seed=seed)

    # create model and variables
    model = Model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    warmup_ratio = 0.95
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=warmup_ratio)
    criterion = nn.MSELoss()
    score = F1Score()

    # create dataloader
    train_set:Dataset = Dataset("train")
    valid_set:Dataset = Dataset("valid")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # create trainer
    trainer = Trainer(model,
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
