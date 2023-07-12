from tqdm.auto import tqdm
import torch
from dataset.customDataset import CustomDataSet
from util.utility import init
from hyperparameters import infer_conf
from model.customModel import CustomModel
from config.configClass import Config
from ckptmanager.manager import CheckPointManager


def main(conf:Config) -> None:
    """Inferencing model base on given config."""

    # device setting
    device = torch.device(conf.device)

    # setup model and other components
    model = CustomModel(conf)                                                # create model
    model.eval()

    # load model from checkpoint if needed
    ckpt_manager = CheckPointManager(conf, model)
    if conf.Load:
        ckpt_manager.load(ckpt_path=conf.load_path, device=device)

    # prepare test dataset and dataloader
    dataset_name = conf.dataset_name
    test_dataset = CustomDataSet(conf, "test", file_name=dataset_name)       # create test dataset

    # start inference
    with torch.no_grad():
        model.eval()

        err_count = 0
        total_count = len(test_dataset) #type:ignore

        for data in tqdm(test_dataset, desc="Inferencing"): #type:ignore
            data.to(device)
            # TODO: add your own inferencing code here

        # TODO: inferencing the result
        print(f"Model: {conf.model_name}, dataset: {conf.dataset_name}")
        print(f"Error: {err_count}/{total_count} = {err_count/total_count*100:0.4f}%")


if __name__ == '__main__':
    #%% print version information
    print(f'Torch version: {torch.__version__}')
    # initialize
    init(infer_conf.seed)

    #%% run main function
    main(infer_conf)
