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
from torch_geometric import nn as tgnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.logs import log_point_cloud_to_wandb
import numpy as np


class AMGNNmodel(pl.LightningModule):
    """Main class of AMGNN. This Pytorch lightning class will deal with the model loading, training, testing, validation
     and logings"""

    def __init__(self, config):
        super().__init__()
        self.configuration = config
        #self.network = NeuralNetwork(self.configuration["input_channels"], self.configuration["hidden_channels"],
        #                             self.configuration["out_channels"], self.configuration['aggregator'],
        #                             self.configuration["number_hidden_layers"])
        self.network = tgnn.models.GIN(self.configuration["input_channels"], self.configuration["hidden_channels"],
                                        self.configuration["number_hidden_layers"],self.configuration["out_channels"])
        self.lr = float(self.configuration["learning_rate"])
        self.batch_size = int(config["batch_size"])
        self.lambda_weight = torch.tensor(config["lambda_parameters"], dtype=torch.float32).view(3, 1)
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """Train loop of the neural network"""
        # Add noise to the data
        y_hat, loss, loss_mse, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train loss mse", loss_mse, batch_size=self.batch_size)
        self.log("train loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("train loss gradient temperature", loss_temp, batch_size=self.batch_size)

        # Log the first test batch results as
        if batch_idx == 0:
            # Save the output
            batch_cpu = batch.detach().to("cpu")
            batch_cpu.y = torch.hstack([batch_cpu.y.detach(), y_hat.detach().to("cpu")])
            train_output_folder = osp.join(self.configuration["raw_data"], "train_output")
            # Create the output folder if it did not exist
            Path(train_output_folder).mkdir(parents=True, exist_ok=True)
            pt_path = osp.join(train_output_folder, f'{batch_idx}.pt')
            torch.save(batch_cpu, pt_path)
            # Transform the pt file as a vtk file that can be read with Pyvista
            read_pt_batch_results(pt_path, self.configuration)
        return loss

    def test_step(self, batch, batch_idx):
        """The test set is NOT used during training, it is ONLY used once the model has been trained to see how the
         model will do in the real-world."""
        y_hat, loss, loss_mse, loss_disp, loss_temp = self._get_preds_loss(batch, return_gradient=True)
        self.log("test loss", loss, batch_size=self.batch_size)
        self.log("test loss mse", loss_mse, batch_size=self.batch_size)
        self.log("test loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("test loss gradient temperature", loss_temp, batch_size=self.batch_size)

        # Save the output
        batch_cpu = batch.to("cpu")
        batch_cpu.y = torch.hstack([batch_cpu.y, y_hat.to("cpu")])
        test_output_folder = osp.join(self.configuration["raw_data"], "test_output")
        # Create the output folder if it did not exist
        Path(test_output_folder).mkdir(parents=True, exist_ok=True)
        pt_path = osp.join(test_output_folder, f'{batch_idx}.pt')
        torch.save(batch_cpu, pt_path)
        # Transform the pt file as a vtk file that can be read with Pyvista
        read_pt_batch_results(pt_path, self.configuration)

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
        y_hat, loss, loss_mse, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("val loss", loss, batch_size=self.batch_size)
        self.log("val loss mse", loss_mse, batch_size=self.batch_size)
        self.log("val loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("val loss gradient temperature", loss_temp, batch_size=self.batch_size)

        # Log the first test batch results as
        if batch_idx == 0:
            batch.y = torch.hstack([batch.y, y_hat])  # Merge y and y_hat
            for graph_id in range(min(batch.num_graphs, 4)):
                graph = batch[graph_id].to("cpu")
                name = graph.file_name
                points = graph.pos * self.configuration["scaling_size"]
                y, y_hat = torch.split(graph.y, 4, dim=1)

                temp_error = (y[:, 0] - y_hat[:, 0]).detach().cpu().numpy()
                # temp_error = temp_error * self.configuration["scaling_temperature"]
                deformation_error = (y[:, 1:] - y_hat[:, 1:]).detach().cpu().numpy()
                deformation_error = np.linalg.norm(deformation_error, axis=1)
                # deformation_error = deformation_error * self.configuration["scaling_deformation"]

                log_point_cloud_to_wandb(name=name + " temperature error",
                                         points=points, value=temp_error,
                                         max_value=0.1,  # self.configuration["scaling_temperature"],
                                         epoch_number=self.current_epoch)

                log_point_cloud_to_wandb(name=name + " deformation error",
                                         points=points, value=deformation_error,
                                         max_value=0.1,  # self.configuration["scaling_size"],
                                         epoch_number=self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': 'train loss'
        }

    def _get_preds_loss(self, batch, return_gradient: bool =False) -> Tuple[torch.Tensor, torch.Tensor]:
        """The train, test and validation function are the same so their are regrouped here.
        Return:
            y_hat: the prediction of the neural network,
            loss: the actual loss of the neural network"""

        # Check no wrong values are used as inputs.
        self.__chech_nan_and_inf(batch.x)
        self.__chech_nan_and_inf(batch.edge_index)
        self.__chech_nan_and_inf(batch.edge_attr)

        y_hat = self.network(x=batch.x, edge_index=batch.edge_index)

        #loss, loss_mse, loss_disp, loss_temp = AMGNN_loss(batch, batch.y, y_hat, detail_loss=True,
        #                                                  lambda_weight=self.lambda_weight)
        loss_function = torch.nn.MSELoss()
        loss= loss_mse= loss_disp= loss_temp = loss_function(batch.y, y_hat)

        return y_hat, loss, loss_mse, loss_disp, loss_temp

    def __chech_nan_and_inf(self, values: Tensor):
        """ Check if there is a Nan of inf value in a tensor.
        Raise an error if so.
        Parameters
        ----------
        values: Tensor
            Tensor to check if there is a Nan of Inf value

        Returns
        -------

        """
        have_nan = torch.isnan(values).any()
        have_inf = torch.isinf(values).any()

        if have_nan:
            raise "Nan found"
        if have_inf:
            raise "Inf found"


class NeuralNetwork(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, aggregator: str, number_hidden: int):
        """ Neural network used to predict the simulation results.

        Graph neural network used to predict the deformation and temperature of nodes using the previous know
        deformation and temperature.

        Parameters
        ----------
        in_channels : int
            Number of features per input node.
        hidden_channels : int
            Size of the hidden layers.
        out_channels : int
            Number of predicted features per node.
        aggregator : str
            Type of aggregator used for the message parsing step (can be mean, max, min...).
        number_hidden : int
            Number of blocks between the first layer and the output layer.
        """
        super().__init__(aggr=aggregator)
        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)

        # Input layer: The encoder
        self.lin = tgnn.models.MLP(in_channels=in_channels, hidden_channels=self.hidden, act="relu",
                                   out_channels=self.hidden, num_layers=3)
        # Output layer: The decoder
        self.lin2 = tgnn.models.MLP(in_channels=self.hidden, hidden_channels=self.hidden, act="relu",
                                    out_channels=out_channels,
                                    num_layers=3)

        dim = self.hidden
        for i in range(self.number_hidden):
            # Add the neural layer to the Message Passing neural network
            setattr(self, f"message_mlp_{i}", tgnn.models.MLP(in_channels=dim * 2, hidden_channels=dim,
                                                              act="gelu", out_channels=dim, num_layers=3, dropout=0.3, )
                    )

    def forward(self, batch: Data) -> Tensor:
        """ Predict the simulation output.

        Using the neural network defined at the creation of this object, the deformation and temperature of this
        simulation step are predicted.

        Parameters
        ----------
        batch: Data
            Graph representing multiple simulation.

        Returns
        -------
        Tensor
            Node tensor representing the simulation state of each node a the predicted time.

        """
        x = batch.x
        edge_index = batch.edge_index
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 4-5: Start propagating messages.

        for i in range(self.number_hidden):
            # Found the Nth layer of the neural network
            layer = getattr(self, f"message_mlp_{i}")
            # Append its results in x_
            message = self.propagate(edge_index, x=x)
            x_ = torch.cat([x, message], dim=1)

            x = layer(x_) + x

        x = self.lin2(x)
        return x

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        """ Compute the message of each edges.

        The message of each edges is calculated as the concatenation of
        \[ x_i , x_i - x_j\]

        Where:

        Parameters
        ----------
        x_i : Tensor
            Features of the receiver node, of shape [Edge number, out_channels].
        x_j : Tensor
            Features of the neighbour node, of shape [Edge number, out_channels].
        model : nn.Sequential
            Model to apply to the message.

        Returns
        -------
        Tensor
            Output of the neural network block.
        """

        # This is an edges convolution
        # https://arxiv.org/abs/1801.07829

        msg = x_j - x_i

        return msg
