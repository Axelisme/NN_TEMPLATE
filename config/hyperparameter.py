
"""define hyperparameters for training and testing """

from config.configClass import Config
import torch

# create config
config = Config(
    project_name = 'NN_TEMPLATE',
    model_name = 'model_1',
    seed = 0,
    output_size = 1,
    device = torch.device('cuda' if torch.cuda.is_available() else 'none'),
    batch_size = 8,
    epochs = 20,
    lr = 0.00001,
    weight_decay = 0.7,
    split_ratio = 0.2,
    SAVE = False,
    WandB = True,
)