
"""A script to train a model on the train dataset."""

import util.utility as ul
import global_var.path as p
from model.model import Model
from config.config import Config
from tester.tester import Tester
from dataset.dataset import DataSet
from trainer.trainer import Trainer
from evaluator.em_score import EMScore
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

def main():
    """Main function of the script."""
    # set random seed
    seed = 1
    ul.set_seed(seed=seed)

    # create model and variables
    model = Model()
    config = Config(
        seed = 0,
        model_name = 'model',
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size = 8,
        epochs = 10,
        lr = 0.001,
        weight_decay = 0.99,
        SAVE = True
    )
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    evaluator = EMScore()

    # create dataloader
    train_set:DataSet = DataSet("train")
    valid_set:DataSet = DataSet("valid")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # create trainer
    trainer = Trainer(model=model, config=config, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    valider = Tester(model=model, config=config, test_loader=valid_loader, evaluators={type(evaluator).__name__:evaluator})

    # start training
    print('Start training model...')
    result = ul.Result()
    for epoch in range(config.epochs):
        print('-'*50)
        print(f'Epoch {epoch+1}/{config.epochs}')
        # train
        train_result:dict = trainer.train()
        # validate
        valid_result:dict = valider.eval()
        # update learning rate
        scheduler.step()
        # print result
        print(f'Train loss: {train_result["train_loss"]:.3f}')
        print("Valid result:")
        for name,score in valid_result.items():
            print(f'{name}: {score:.3f}')
        # store result
        result.add(train_result)
        result.add(valid_result)
    print('Finished training model.')

    # save model
    post_fix = f"loss_{result['train_loss'][-1]:.3f}_score_{result[type(evaluator).__name__][-1]:.3f}"
    SAVE_MODEL_PATH = p.SAVED_MODELS_DIR + f"/{config.model_name}_{post_fix}.pt"
    if config.SAVE:
        print(f'Saving model to {SAVE_MODEL_PATH}')
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print('Finished saving model.')


if __name__ == '__main__':
    # print information
    print(f'Torch version: {torch.__version__}')
    print(f'Data discription: ')
    DataSet.discription()

    # start training
    main()
