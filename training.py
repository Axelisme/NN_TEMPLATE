
"""A script to train a model on the train dataset."""

import config.path as p
import util.utility as ul
from model.model import Model
from dataset.dataset import DataSet
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
    ul.set_seed(0)

    # create model and variables
    model = Model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    batch_size = 8
    learning_rate = 0.001
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    warmup_ratio = 0.999
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=warmup_ratio)
    criterion = nn.CrossEntropyLoss()
    score = F1Score()
    SAVE = False
    PLOT = False

    # create dataloader
    train_set:DataSet = DataSet("train")
    valid_set:DataSet = DataSet("valid")
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
    print('Start training model...')
    result = trainer.train()
    print('Finished training model.')

    # save model
    post_fix = f"loss_{result['valid_loss'][-1]:.3f}_score_{result['valid_score'][-1]:.3f}"
    SAVE_MODEL_PATH = p.SAVED_MODELS_DIR + f"/model_{post_fix}.pt"
    SAVE_RESULT_PATH = p.TRAIN_RESULTS_DIR + f"/result_{post_fix}.csv"
    if SAVE:
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        result.save(SAVE_RESULT_PATH)

    # plot results
    if PLOT:
        result.plot()

if __name__ == '__main__':
    # print information
    print(f'Torch version: {torch.__version__}')
    print(f'Data discription: ')
    DataSet.discription()

    # start training
    main()
