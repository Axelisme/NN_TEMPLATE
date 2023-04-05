
"""A script to train a model on the train dataset."""

import util.utility as ul
import global_var.path as p
from model.model import MyModel
from config.config import Config
from tester.tester import Tester
from loss.loss import Loss
from dataset.dataset import MyDataSet
from trainer.trainer import Trainer
from evaluator.em_score import EMScore
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils import data
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

def main():
    """Main function of the script."""
    # create config
    config = Config(
        project_name = 'Template_NN',
        model_name = 'test',
        seed = 0,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size = 16,
        epochs = 100,
        lr = 0.001,
        weight_decay = 0.99,
        SAVE = False,
        WANDB = True
    )

    # initialize wandb
    if config.WANDB: # type: ignore[attr]
        wandb.init(project=config.project_name, config=config.data)
        wandb.run.name = f"{config.model_name}_{wandb.run.id}"  # type: ignore[attr]


    # set random seed
    ul.set_seed(seed=config.seed)

    # create loss, optimizer, scheduler, criterion, evaluator
    model = MyModel()
    #model.load_state_dict(torch.load(p.SAVED_MODELS_DIR + '/QINN_test_loss_1.477_score_0.948.pt'))
    dataset = MyDataSet
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    evaluator = EMScore()

    # create dataloader
    dataset.load_data()
    train_set:data.Dataset = dataset("train")
    valid_set:data.Dataset = dataset("valid")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # create trainer
    trainer = Trainer(model=model, config=config, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    valider = Tester(model=model, config=config, test_loader=valid_loader)
    valider.add_evaluator(evaluator)

    # start training
    config.WANDB and wandb.watch(model, criterion=criterion, log='all', log_freq=100) # type: ignore[attr]
    print('Start training model...')
    for epoch in tqdm(range(config.epochs),leave=True):
        print('-'*50)
        print(f'Epoch {epoch+1}/{config.epochs}')
        # train
        train_result:dict = trainer.train()
        # validate
        valid_result:dict = valider.eval()
        # update learning rate
        scheduler.step()
        # print result
        print(f'Train loss: {train_result["train_loss"] :.3f}')
        print("Valid result:")
        for name,score in valid_result.items():
            print(f'{name}: {score:.3f}')
        # store result
        train_result.update(valid_result)
        config.WANDB and wandb.log(train_result, step=epoch) # type: ignore[attr]
    print('Finished training model.')

    # save model
    SAVE_MODEL_PATH = p.SAVED_MODELS_DIR + f"/{config.model_name}.pt"
    if config.SAVE:
        print(f'Saving model to {SAVE_MODEL_PATH}')
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        config.WanbB and wandb.save(SAVE_MODEL_PATH) # type: ignore[attr]
        print('Finished saving model.')

    config.WANDB and wandb.finish() # type: ignore[attr]

if __name__ == '__main__':
    # print information
    print(f'Torch version: {torch.__version__}')
    print(f'Data discription: ')
    MyDataSet.discription()

    # start training
    main()

