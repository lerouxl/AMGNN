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
import torch_geometric.transforms as T
import logging
from utils.logs import init_logger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from datetime import datetime


def run():
    """Entry strips to train an test AMGNN model on data.
    The configuration of AMGNN and the training is defined in the *configs* folder.

    Training, validation and testing results are saved using the Wandb library.
    The dataset is randomly split into 3 sets (train, test and validation)
    """
    # Set the seed of torch, numpy and random.
    pl.seed_everything(51, workers=True)
    # Configure a text logger
    init_logger('logs.log')
    log = logging.getLogger(__name__)

    # Initialise wandb
    configuration = read_config(Path("configs"))
    name = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) #  + "_" + configuration["model_name"]

    # Try to log the model on Wandb
    if configuration["offline"]:
        # If the run is offline, log_model cannot be true as this is an invalid configuration"
        # since model checkpoints cannot be uploaded in offline mode
        log_model = False
    else:
        log_model = True

    wandb_logger = WandbLogger(project="AMGNN", config=configuration, name=name, offline=configuration["offline"],
                               notes=configuration["notes"], tags=configuration["tags"], log_model=log_model)
    log.info("Configuration loaded")

    # Access all hyperparameters values through wandb.config
    configuration = dict(wandb.config)
    print(configuration)

    # Add model name to the tags
    wandb_logger.experiment.tags = wandb_logger.experiment.tags + (configuration["model_name"],)
    # The name update is moved there as the model_name variable can be updated by a sweep
    wandb_logger.experiment.name = name + "_" + configuration["model_name"]
    # Create the deep learning model
    model = AMGNNmodel(configuration)
    wandb_logger.watch(model.network, log="all")
    log.info("AMGNN model created")

    # Create the dataset
    data_path = Path(configuration["raw_data"])
    # , T.RadiusGraph(r=2/ configuration["scaling_size"])
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.Distance()])
    arc_dataset = ARCDataset(data_path, transform=transform)
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

    # Set the data loader into the model
    model.set_train_dataloader(train_loader)
    model.set_test_dataloader(test_loader)
    model.set_val_dataloader(validation_loader)

    # Create the pytorch lightning trainer.
    log.info("Start model training")
    # Create a trained run the model on the GPU, with a wandb logger, saving the best 2 models in the checkpoints dir
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{name}/",
                                          save_top_k=1, monitor="val loss",
                                          filename='amgnn-{epoch:02d}')
    lr_callback = LearningRateMonitor(logging_interval="step")
    stocha_weight_ave = StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=20 )

    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         logger=wandb_logger,
                         auto_lr_find=True,
                         callbacks=[checkpoint_callback, lr_callback, stocha_weight_ave],
                         default_root_dir=f"checkpoints/{name}/",
                         max_epochs=configuration["max_epochs"],
                         accumulate_grad_batches=int(configuration["accumulate_grad_batches"]),
                         auto_scale_batch_size="binsearch",
                         check_val_every_n_epoch=1,
                         log_every_n_steps= 50,
                         gradient_clip_val=0.8
                         )

    # From https://pytorch-lightning.readthedocs.io/en/1.4.5/advanced/lr_finder.html
    # Found the optimal learning rate:
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model)
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    # update hparams of the model
    model.lr = new_lr
    print(f"New learning rate found: {new_lr}")

    trainer.fit(model)
    #trainer.fit(model, train_loader, validation_loader)
    log.info("End model training")

    # Test the model on unseen data with the best model
    best_model_ = checkpoint_callback.best_model_path
    log.info(f"Start model testing with {str(best_model_)}")
    trainer.test(dataloaders=test_loader, ckpt_path=best_model_)
    log.info("End model testing")

    # Display the test dataset results in vtk files (can be open with Paraview)
    # log.info("Generate visualisation of the results")
    # display_dataset(best_model_, test_dataset, configuration, "Test_dataset_results")

    log.info("END")


if __name__ == "__main__":
    run()
