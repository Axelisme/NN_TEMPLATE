import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from dataset.dataset import DataSet
from util.utility import set_seed, get_cuda
from util.io import logit, plot_confusion_matrix
from util.checkpoint import load_checkpoint
from util.tool import clear_folder
from hyperparameter import *
from model.customModel import CustomModel
from config.configClass import Config


config = Config(
    device = get_cuda(),
    batch_size = 1,
)
config.update(base_config)

#config.load_path = path.join(SAVED_MODELS_DIR, config.model_name, f"checkpoint_{config.model_name}_20.pt")


@logit(LOG_CONSOLE)
def main(conf:Config) -> None:
    """Main function of the script."""

    # create model and load
    model = CustomModel(conf)                                                # create model
    load_checkpoint(model, checkpoint_path=conf.load_path)                     # load model
    model.eval().to(conf.device)                                             # set model to eval mode

    # prepare images
    dataset_name = "dataset_all.hdf5"
    test_dataset = DataSet(conf, "test", dataset_name=dataset_name)          # create test dataset
    label_names = test_dataset.label_names
    for id,name in enumerate(label_names):
        print(f"{id}:{name}", end=' ')
    print()

    # start inference
    post_fix = "color"
    SAVE_INFER_DIR = os.path.join(INFER_EX_DIR, conf.model_name, dataset_name)
    ERROR_DIR = os.path.join(SAVE_INFER_DIR, f"error_image_{post_fix}")
    clear_folder(ERROR_DIR)
    total_count = 0
    err_count = 0
    misclassified = np.zeros((conf.output_size,conf.output_size),dtype=np.float32)
    with torch.no_grad():
        for input, label  in tqdm(test_dataset, desc="Inferencing"): #type:ignore
            total_count += 1
            output = model(input.to(conf.device).unsqueeze(0)).squeeze(0).cpu()
            misclassified[label] += output.squeeze(0).numpy()
            pred = output.argmax(dim=0).item()
            if label != pred:
                err_count += 1
                label_name = label_names[label]
                pred_name = label_names[pred]
                #print(f"Error: pred:{pred_name} != label:{label_name}.")
                file_name = f"label_{label_name}_pred_{pred_name}_{err_count}.jpg"
                input_img = transforms.ToPILImage()(input)
                input_img.save(os.path.join(ERROR_DIR, file_name))

    print(f"Model: {conf.model_name}, dataset: {dataset_name}, data: {post_fix}")
    print(f"Error: {err_count}/{total_count} = {err_count/total_count*100:0.4f}%")

    # plot confusion matrix
    title = f"model: {conf.model_name}, dataset: {dataset_name}, data: {post_fix}"
    print(f"Plotting {title}...")
    CM_PATH = os.path.join(SAVE_INFER_DIR, f"CM_{post_fix}_err_{err_count/total_count*100:0.1f}.png")
    plot_confusion_matrix(cm=misclassified, class_names=label_names, path=CM_PATH, title=title, normalize=True)


def init(conf:Config) -> None:
    """Initialize the script."""
    # create directory
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    # set float32 matmul precision
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.set_float32_matmul_precision('medium')
    # set random seed
    set_seed(seed=conf.seed, cudnn_benchmark=True)


if __name__ == '__main__':
    # print version information
    print(f'Torch version: {torch.__version__}')
    init(config)

    # run main function
    main(config)
