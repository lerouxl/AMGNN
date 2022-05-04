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


def run():
    """Train an test AMGNN model on the data"""
    # Initialise wandb
    configuration = read_config(Path("configs"))

    wandb_logger = WandbLogger(project="AMGNN", log_model=True, mode="offline", config=configuration)

    # Access all hyperparameters values through wandb.config
    configuration = wandb.config
    print(configuration)

    # Create the deep learning model
    model = AMGNNmodel(configuration)

    # Create the dataset
    data_path = Path(configuration["raw_data"])
    arc_dataset = ARCDataset(data_path)

    # Split the dataset in 3 subset, the train, validation and test dataset.
    dataset_size = len(arc_dataset)
    train_size = int(dataset_size * 0.80)  # 80% of the dataset
    validation_size = int(dataset_size * 0.15)  # 15% of the dataset
    test_size = dataset_size - train_size - validation_size  # 5% of the dataset
    # Split the dataset with a repeatable way (the random generator is fixed)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(arc_dataset,
                                                                                    [train_size, validation_size,
                                                                                     test_size],
                                                                                    generator=torch.Generator().manual_seed(
                                                                                        51))

    # Create the dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)

    # Create the pytorch lightning trainer.
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=wandb_logger)
    trainer.fit(model, train_loader, validation_loader)
    trainer.test(dataloaders=test_loader)

    # Display the test dataset results in vtk files (can be open with Paraview)
    display_dataset(model, test_dataset, configuration)


if __name__ == "__main__":
    run()
