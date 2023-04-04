
"""A script to run inference on a trained model."""

from sys import argv
import global_var.path as p
import util.utility as ul
from model.model import Model
from dataset.dataset import DataSet
from config.config import Config
from tester.tester import Tester
from evaluator.em_score import EMScore
import torch
from torch import nn
from torch.utils.data import DataLoader

def main(model_path: str) -> None:
    """Main function of the script."""
    # set random seed
    seed = 0
    ul.set_seed(seed)

    # create model and variables
    model = Model()
    config = Config(
        seed=seed,
        model_name='model',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=8,
        SAVE=False
    )
    evaluator = EMScore()

    # load model
    model.load_state_dict(torch.load(model_path, map_location=config.device))

    # create dataloader
    test_set:DataSet = DataSet("test")
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # create trainer
    tester = Tester(model=model, config=config, test_loader=test_loader)
    tester.add_evaluator(name=None, evaluator=evaluator)

    # start testing
    result:dict = tester.eval()

    # print result
    print("Result:")
    for name,score in result.items():
        print(f'{name}: {score:.3f}')

if __name__ == '__main__':
    # print version information
    print(f'Torch version: {torch.__version__}')
    print(f'Data discription: ')
    DataSet.discription()

    # find the save model path
    save_model_path = ''
    if len(argv) == 2:
        main(argv[1])
    else:
        import os
        all_model = os.listdir(p.SAVED_MODELS_DIR)
        if '.gitkeep' in all_model:
            all_model.remove('.gitkeep')
        if len(all_model) == 0:
            raise Exception("No model found in saved_models directory.")
        save_model_path = os.path.join(p.SAVED_MODELS_DIR,all_model[0]) 

    # run main function
    main(save_model_path)