import torch
import wandb
from utils.config import read_config
from pathlib import Path
from model.amgnn import AMGNNmodel
from dataloader.arc_dataset import ARCDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from utils.visualise import display_dataset
import logging
from utils.logs import init_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from torch import optim, nn
from typing import Tuple
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
import os.path as osp
from utils.loss_function import AMGNN_loss
from utils.visualise import read_pt_batch_results
from pathlib import Path
from torch_geometric.nn import MLP
import wandb
from utils.logs import log_point_cloud_to_wandb
import numpy as np


def eval(model_path):
    """Will generate visualition file from the train dataset.
    """
    torch.no_grad()
    # Configure a text logger
    init_logger('logs.log')
    log = logging.getLogger(__name__)

    # Initialise wandb
    configuration = read_config(Path("configs"))
    name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    wandb_logger = WandbLogger(project="AMGNN", config=configuration, name=name, offline=True)
    log.info("Configuration loaded")

    # Access all hyperparameters values through wandb.config
    configuration = dict(wandb.config)
    print(configuration)

    # Create the deep learning model
    model = AMGNNmodel(configuration)
    model = model.load_from_checkpoint(model_path)
    model.eval()
    log.info("AMGNN model created")

    # Create the dataset
    data_path = Path(configuration["raw_data"])
    arc_dataset = ARCDataset(data_path)
    log.info(f"ARCDataset created from {str(data_path)}")

    # Split the dataset in 3 subset, the train, validation and test dataset.
    dataset_size = len(arc_dataset)
    train_size = int(dataset_size * 0.80)  # 80% of the dataset
    validation_size = int(dataset_size * 0.15)  # 15% of the dataset
    test_size = dataset_size - train_size - validation_size  # 5% of the dataset
    log.info(f"The dataset will be divided in 3 sub set of size {train_size=} {validation_size=} {test_size=} ")

    # Split the dataset with a repeatable way (the random generator is fixed)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(arc_dataset,
                                                                                    [train_size, validation_size,
                                                                                     test_size],
                                                                                    generator=torch.Generator().manual_seed(
                                                                                        51))
    log.info("Subset created")

    # Create the dataloader
    batch_size = int(configuration["batch_size"])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    log.info(f"Dataloader created with {batch_size=}")

    batch_idx = 0
    for batch in iter(train_loader):
        y_hat, loss, loss_mse, loss_disp, loss_temp = model._get_preds_loss(batch)
        # Save the output
        batch_cpu = batch.to("cpu")
        batch_cpu.y = torch.hstack([batch_cpu.y, y_hat.to("cpu")])
        test_output_folder = osp.join(model.configuration["raw_data"], "train_output")
        # Create the output folder if it did not exist
        Path(test_output_folder).mkdir(parents=True, exist_ok=True)
        pt_path = osp.join(test_output_folder, f'{batch_idx}.pt')
        torch.save(batch_cpu.detach(), pt_path)
        # Transform the pt file as a vtk file that can be read with Pyvista
        read_pt_batch_results(pt_path, model.configuration)
        batch_idx = batch_idx +1

        if batch_idx > 5:
            break


if __name__ == "__main__":
    model_path = r"E:\Leopold\Chapitre 6 - AMGNN\AMGNN\checkpoints\2022_10_29_20_04_26\epoch=24-step=2500.ckpt"
    eval(model_path)