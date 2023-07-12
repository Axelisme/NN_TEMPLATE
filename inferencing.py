from tqdm.auto import tqdm
import torch
from dataset.customDataset import CustomDataSet
from util.utility import init
from util.checkpoint import load_checkpoint
from hyperparameters import *
from model.customModel import CustomModel
from config.configClass import Config


def main(conf:Config) -> None:
    """Main function of the script."""

    # device setting
    device = torch.device(conf.device)

    # create model and load
    model = CustomModel(conf)                                                # create model
    load_checkpoint(model, conf.load_path, device=device)                    # load model
    model.eval()                                                             # set model to eval mode

    # prepare test dataset and dataloader
    test_dataset = CustomDataSet(conf, "test", file_name=conf.dataset_name)    # create test dataset

    # start inference
    with torch.no_grad():
        err_count = 0
        total_count = len(test_dataset) #type:ignore

        for data in tqdm(test_dataset, desc="Inferencing"): #type:ignore
            data.to(device)
            # TODO: add your own inferencing code here

        # TODO: inferencing the result
        print(f"Model: {conf.model_name}, dataset: {conf.dataset_name}")
        print(f"Error: {err_count}/{total_count} = {err_count/total_count*100:0.4f}%")


if __name__ == '__main__':
    # print version information
    print(f'Torch version: {torch.__version__}')
    init(infer_conf.seed)

    # run main function
    main(infer_conf)
