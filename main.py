import torch
import wandb
from utils.config import read_config
from pathlib import Path
from model.amgnn import AMGNNmodel
from dataloader.arc_dataset import ARCDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from utils.visualise import display_X_Y_Y_hat_data

def run():
    """Train an test AMGNN model on the data"""
    # Initialise wandb
    configuration = read_config(Path("configs"))

    wandb_logger = WandbLogger(project="AMGNN", log_model=True, mode="offline", config=configuration)
    wandb.init(mode="offline", config=configuration)

    # Access all hyperparameters values through wandb.config
    configuration = wandb.config
    print(configuration)

    model = AMGNNmodel(configuration)
    DATA_PATH = Path("data/light")
    arc_dataset = ARCDataset(DATA_PATH)
    arc_loader = DataLoader(dataset=arc_dataset , batch_size= 2, shuffle= True)

    trainer = pl.Trainer(limit_train_batches=10, max_epochs=1, accelerator="gpu", devices=1, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=arc_loader)

    with torch.no_grad():
        model.eval()
        batch = arc_dataset[0]
        y_hat = model.network(batch)
        display_X_Y_Y_hat_data(pos=batch.pos.numpy(), X=batch.x.numpy(),
                               Y=batch.y.numpy(), Y_hat=y_hat.numpy(), config=configuration)


if __name__ == "__main__":
    run()
