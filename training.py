
"""A script to train a model on the train dataset."""

#%%
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
    weight_decay = 0.99,
    split_ratio = 0.2,
    SAVE = False,
    WandB = False
)

#%%
def main():
    """Main function of the script."""

    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')

    # set random seed
    ul.set_seed(seed=config.seed)

    # create loss, optimizer, scheduler, criterion, evaluator
    model = Model(config)
    #model.load_state_dict(torch.load(p.SAVED_MODELS_DIR + '/QINN_test.pt'))
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    evaluator = cf.MulticlassAccuracy(num_classes=config.output_size,average='micro', validate_args=True)
    evaluator2 = cf.MulticlassF1Score(num_classes=config.output_size,average='micro', validate_args=True)

    # create dataloader
    DataSet.load_data(config)
    train_set:DataSet = DataSet("train")
    valid_set:DataSet = DataSet("valid")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # create trainer
    trainer = Trainer(model=model, config=config, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    valider = Tester(model=model, config=config, test_loader=valid_loader)
    valider.add_evaluator(evaluator)

    # start training
    config.WandB and wandb.watch(model, criterion=criterion, log='all', log_freq=500) # type: ignore
    print('Start training model...')
    for epoch in tqdm(range(config.epochs)):
        print('-'*50)
        print(f'Epoch {epoch+1}/{config.epochs}')
        # train
        train_result:dict = trainer.train()
        # validate
        valid_result:dict = valider.eval()
        # update learning rate
        scheduler.step()
        # print result
        print(f'Train loss: {train_result["train_loss"]}')
        print("Valid result:")
        for name,score in valid_result.items():
            print(f'{name}: {score}')
        # store result
        train_result.update(valid_result)
        config.WandB and wandb.log(train_result, step=epoch) # type: ignore
    print('Finished training model.')

    # save model
    SAVE_MODEL_PATH = p.SAVED_MODELS_DIR + f"/{config.model_name}.pt"
    if config.SAVE:
        print(f'Saving model to {SAVE_MODEL_PATH}')
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        config.WandB and wandb.save(SAVE_MODEL_PATH) # type: ignore
        print('Finished saving model.')


if __name__ == '__main__':
    # print information
    print(f'Torch version: {torch.__version__}')

    #%% initialize wandb
    if config.WandB:
        wandb.init(project=config.project_name, config=config.data)
        wandb.run.name = f"{config.model_name}_{wandb.run.id}" # type: ignore

    #%% start training
    main()

    #%% finish wandb
    config.WandB and wandb.finish() # type: ignore

