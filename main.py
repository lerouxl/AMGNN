import wandb
from utils.config import read_config
from pathlib import Path
from model.amgnn import model


def run():
    """Train an test AMGNN model on the data"""
    # Initialise wandb
    configuration = read_config(Path("configs"))
    wandb.init(mode="offline", config=configuration)

    # Access all hyperparameters values through wandb.config
    configuration = wandb.config
    print(configuration)

    amgnn = model()
    amgnn.load_data()
    amgnn.build()
    amgnn.train()
    amgnn.validate()


if __name__ == "__main__":
    run()
