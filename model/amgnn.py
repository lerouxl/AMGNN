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
# from torch_geometric.nn import MLP
from model.operations import MLP
from torch_geometric.nn.aggr import LSTMAggregation, MeanAggregation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from utils.logs import log_point_cloud_to_wandb
import numpy as np
from model.simple_mlp import SimpleMlp, DoubleHeadSimpleMlp
from model.simple_gnn import SimpleGnn, DoubleHeadSimpleGnn
from model.simple_conv import SimpleSAGEConv, DoubleHeadSimpleSAGEConv
from torch_geometric.loader import DataLoader


def check_tensors(tensor: Tensor):
    """ Check if a tensor is valid.

    Check if a tensor contain infinite values, NaN values or value superior to 1.
    Raise an error if so.

    Parameters
    ----------
    tensor: Tensor
        The tensor to check.
    """
    assert (tensor.max() <= 1), f"A value superior to 1 was found ({str(int(tensor.max()))}) in columns {str((tensor > 1.0).nonzero(as_tuple=True)[1].unique())}"
    assert ( bool(torch.isnan(tensor).any()) is False)
    assert ( bool(torch.isinf(tensor).any()) is False)
    assert ( bool(torch.isneginf(tensor).any()) is False)

    #for i in range(22):
    # print(f"{i} : max val {float(tensor[:,i].max())}")


class AMGNNmodel(pl.LightningModule):
    """Main class of AMGNN dealing with the model creation, training, testing, validation and logins

    This class is a lightning module dealing with AMGNN. This should be considered as the main entry points.
    """

    def __init__(self, config: dict):
        """ Define AMGNN and it's training parameters.

        Using a config dictionary to initialise it.

        Example
        -------
        AMGNN can be trained as following:

            from dataloader.arc_dataset import ARCDataset
            from torch_geometric.loader import DataLoader
            import pytorch_lightning as pl

            configuration = dict("input_channels": 22,
                                 "out_channels": 4,
                                 "number_hidden_layers": 5,
                                 "learning_rate": 0.0001,
                                 "batch_size": 10,
                                 "lambda_parameters": [1,1,1])
            model = AMGNNmodel(configuration)
            dataset = ARCDataset("dataset")
            data_loader = DataLoader(dataset=dataset, batch_size=10)

            trainer = pl.Trainer()
            trainer.fit(model, data_loader)

        AMGNN can be tested as following:

            best_model_ = # Load best model
            trainer.test(dataloaders=test_data_loader, ckpt_path=best_model_)

        Parameters
        ----------
        config: dict
            dictionary containing the description of the model and training parameters.
        """
        super().__init__()
        self.configuration = config
        self.network = self.get_ai_model()
        self.lr = float(self.configuration["learning_rate"])
        self.batch_size = int(config["batch_size"])
        self.lambda_weight = torch.tensor(config["lambda_parameters"], dtype=torch.float32).view(3, 1)
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.save_hyperparameters()
        self.use_ReduceLROnPlateau = True

        # Define the dataloader to none, they can be loader with the set_train_dataloader function
        self._train_dataloader = None
        self._test_dataloader = None
        self._val_dataloader = None

    def set_train_dataloader(self, train_dataloader: DataLoader) -> None:
        """ Load the train dataloader to this class

        Loading the dataloader allow the use of lr rate finder.

        Parameters
        ----------
        train_dataloader: DataLoader
            The train dataloader.
        """
        self._train_dataloader = train_dataloader

    def train_dataloader(self) -> DataLoader:
        dl = self._train_dataloader
        if dl is not None:
            return dl
        else:
            raise AttributeError("Data loader not configured")

    def set_val_dataloader(self, val_dataloader: DataLoader) -> None:
        """ Load the val dataloader to this class

        Loading the dataloader allow the use of lr rate finder.

        Parameters
        ----------
        val_dataloader: DataLoader
            The train dataloader.
        """
        self._val_dataloader = val_dataloader

    def val_dataloader(self) -> DataLoader:
        dl = self._val_dataloader
        if dl is not None:
            return dl
        else:
            raise AttributeError("Data loader not configured")

    def set_test_dataloader(self, test_dataloader: DataLoader) -> None:
        """ Load the test dataloader to this class

        Loading the dataloader allow the use of lr rate finder.

        Parameters
        ----------
        test_dataloader: DataLoader
            The train dataloader.
        """
        self._test_dataloader = test_dataloader

    def test_dataloader(self) -> DataLoader:
        dl = self._test_dataloader
        if dl is not None:
            return dl
        else:
            raise AttributeError("Data loader not configured")

    def get_ai_model(self) -> DoubleHeadSimpleSAGEConv:
        """Get the deep learning model.

        Using the parameter `model_name`, this class is loading, and configuring the neural network.
        The available GNN are:

        - `simple_mlp`: A simple MLP with no message passing (not taking advantage of the graph structure)
        - `simple_gnn`: A simple GNN using two MLP for the encoding and decoding phase.

        Returns
        -------
        MessagePassing:
            The torch geometric neural network.
        """
        model_name = self.configuration["model_name"]

        if model_name == "simple_mlp":
            return SimpleMlp(in_channels=self.configuration["input_channels"],
                             hidden_channels=self.configuration["hidden_channels"],
                             out_channels=self.configuration["out_channels"],
                             number_hidden=self.configuration["number_hidden_layers"])
        elif model_name == "double_head_simple_mlp":
            return DoubleHeadSimpleMlp(in_channels=self.configuration["input_channels"],
                                       hidden_channels=self.configuration["hidden_channels"],
                                       out_channels=self.configuration["out_channels"],
                                       number_hidden=self.configuration["number_hidden_layers"])
        elif model_name == "simple_gnn":
            return SimpleGnn(in_channels=self.configuration["input_channels"],
                             hidden_channels=self.configuration["hidden_channels"],
                             out_channels=self.configuration["out_channels"],
                             number_hidden=self.configuration["number_hidden_layers"],
                             aggregator=self.configuration["aggregator"])
        elif model_name == "double_head_simple_gnn":
            return DoubleHeadSimpleGnn(in_channels=self.configuration["input_channels"],
                                       hidden_channels=self.configuration["hidden_channels"],
                                       out_channels=self.configuration["out_channels"],
                                       number_hidden=self.configuration["number_hidden_layers"],
                                       aggregator=self.configuration["aggregator"])
        elif model_name == "simple_conv":
            return SimpleSAGEConv(in_channels=self.configuration["input_channels"],
                                  hidden_channels=self.configuration["hidden_channels"],
                                  out_channels=self.configuration["out_channels"],
                                  number_hidden=self.configuration["number_hidden_layers"])

        elif model_name == "double_head_simple_conv":
            return DoubleHeadSimpleSAGEConv(in_channels=self.configuration["input_channels"],
                                  hidden_channels=self.configuration["hidden_channels"],
                                  out_channels=self.configuration["out_channels"],
                                  number_hidden=self.configuration["number_hidden_layers"])
        else:
            raise f"{str(model_name)} is not a valid model name."

    def training_step(self, batch: Batch, batch_idx: int):
        """ Definition of the training loop.

        Will log at each step the losses of the networks.

        Parameters
        ----------
        batch: Batch
            Training batch containing the graph.
        batch_idx: int
            id of the batch

        Returns
        -------
        None
        """
        _, loss, loss_mse, loss_disp, loss_temp = self.get_preds_loss(batch)
        self.log("train loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train loss mse", loss_mse, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train loss gradient displacement", loss_disp, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train loss gradient temperature", loss_temp, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: Batch, batch_idx: int):
        """ Definition of the testing loop.

        The test set is *not used* during training, it is *only* used once the model has been trained to see how the
        model will do in the real-world with unseen data. Losses results are logged and results file are created to in
        the `test_output` folder where the graph with there prediction and label can be read with the software Paraview.

        Parameters
        ----------
        batch: Batch
            Training batch containing the graph.
        batch_idx: int
            id of the batch

        Returns
        -------
        None
        """
        y_hat, loss, loss_mse, loss_disp, loss_temp = self.get_preds_loss(batch)
        self.log("test loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test loss mse", loss_mse, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test loss gradient displacement", loss_disp, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test loss gradient temperature", loss_temp, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)

        # Save the output
        batch_cpu = batch.to("cpu")
        batch_cpu.y = torch.hstack([batch_cpu.y, y_hat.to("cpu")])
        test_output_folder = osp.join(self.configuration["raw_data"], f"{wandb.run.name} test_output")
        # Create the output folder if it did not exist
        Path(test_output_folder).mkdir(parents=True, exist_ok=True)
        pt_path = osp.join(test_output_folder, f'{batch_idx}.pt')
        torch.save(batch_cpu, pt_path)
        # Transform the pt file as a vtk file that can be read with Pyvista
        if batch_idx == 1:
            read_pt_batch_results(pt_path, self.configuration)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """ Validate the model.

        During training, it’s common practice to use a small portion of the train split to determine when the model
        has finished training.As a rule of thumb, we use 20% of the training set as the validation set.
        This number varies from dataset to dataset.

        Parameters
        ----------
        batch: Batch
            Batch of data made of graphs. This batch will contain m graph, there nodes features and target.
        batch_idx: int
            Id of the actual batch
        """
        y_hat, loss, loss_mse, loss_disp, loss_temp = self.get_preds_loss(batch)
        self.log("val loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val loss mse", loss_mse, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val loss gradient displacement", loss_disp, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val loss gradient temperature", loss_temp, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """ Configure the optimizer.

        Returns
        -------
        dictionary:
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train loss"
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if self.use_ReduceLROnPlateau:
            print("Use ReduceLROnPlateau")
            scheduler = ReduceLROnPlateau(optimizer)
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "train loss_epoch"
                    }
        else:
            print("Do not use ReduceLROnPlateau")
            return optimizer

    def get_preds_loss(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Use the network to make a prediction from the batch and compute the loss.

        The train, test and validation function are using the same network so the results prediction and loss
        calculation are merged here.

        The `loss` computation is described in `AMGNN_loss` but can be resumed as:
        \[ loss = = \lambda_1 \cdot Loss_{mse} + \lambda_2 \cdot Loss_{gradient deformation} +
         \lambda_3 \cdot Loss_{gradient temperature}\]
        Where the \(\lambda\) are defined in the `self.init` with the input dictionary key `lambda_parameters`

        Parameters
        ----------
        batch: Batch
            Dataset batch.

        Returns
        -------
        y_hat: The output of the neural network.
        loss: The actual loss of the neural network.
        loss_mse: The mse loss of the neural network.
        loss_disp: The gradient loss of the displacement.
        loss_temp: The gradient loss of the temperature.
        """
        # Check input
        check_tensors(batch.x)

        # Output of the network
        y_hat = self.network(batch)

        y = batch.y
        # Check variable
        check_tensors(y)

        # Compute the losses
        loss, loss_mse, loss_disp, loss_temp = AMGNN_loss(batch, y, y_hat, detail_loss=True,
                                                          lambda_weight=self.lambda_weight)

        return y_hat, loss, loss_mse, loss_disp, loss_temp
