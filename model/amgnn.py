from torch import optim, nn
from typing import Tuple
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import pytorch_lightning as pl
import os.path as osp
from utils.loss_function import AMGNN_loss
from utils.visualise import read_pt_batch_results
from pathlib import Path
from torch_geometric.nn import MLP


class AMGNNmodel(pl.LightningModule):
    """Main class of AMGNN. This Pytorch lightning class will deal with the model loading, training, testing, validation
     and logings"""

    def __init__(self, config):
        super().__init__()
        self.configuration = config
        self.network = NeuralNetwork(self.configuration["input_channels"], self.configuration["hidden_channels"],
                                     self.configuration["out_channels"], self.configuration['aggregator'],
                                     self.configuration["number_hidden_layers"])
        self.lr = float(self.configuration["learning_rate"])
        self.batch_size = int(config["batch_size"])
        self.lambda_weight = torch.tensor(config["lambda_parameters"], dtype=torch.float32).view(3,1)
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """Train loop of the neural network"""
        _, loss, loss_mse, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train loss mse", loss_mse, batch_size=self.batch_size)
        self.log("train loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("train loss gradient temperature", loss_temp, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """The test set is NOT used during training, it is ONLY used once the model has been trained to see how the
         model will do in the real-world."""
        y_hat, loss, loss_mse, loss_disp, loss_temp = self._get_preds_loss(batch)
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
        torch.save(batch_cpu,pt_path)
        # Transform the pt file as a vtk file that can be read with Pyvista
        read_pt_batch_results(pt_path, self.configuration)


    def validation_step(self, batch, batch_idx):
        """During training, it’s common practice to use a small portion of the train split to determine when the model
         has finished training.As a rule of thumb, we use 20% of the training set as the validation set.
         This number varies from dataset to dataset."""
        _, loss, loss_mse, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("val loss", loss, batch_size=self.batch_size)
        self.log("val loss mse", loss_mse, batch_size=self.batch_size)
        self.log("val loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("val loss gradient temperature", loss_temp, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _get_preds_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """The train, test and validation function are the same so their are regrouped here.
        Return:
            y_hat: the prediction of the neural network,
            loss: the actual loss of the neural network"""

        y_hat = self.network(batch)

        loss, loss_mse, loss_disp, loss_temp = AMGNN_loss(batch, batch.y, y_hat, detail_loss=True,
                                                          lambda_weight=self.lambda_weight)

        return y_hat, loss, loss_mse, loss_disp, loss_temp


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

        # Input layer
        self.lin = nn.Linear(in_channels, self.hidden)
        # Output layer
        self.lin2 = nn.Linear(self.hidden, self.hidden)

        self.temperature = MLP(in_channels=self.hidden, hidden_channels=self.hidden,
                               out_channels=1, num_layers=3, dropout=0.3, )
        self.deformation = MLP(in_channels=self.hidden, hidden_channels=self.hidden,
                               out_channels=3, num_layers=3, dropout=0.3, )

        dim = self.hidden
        for i in range(self.number_hidden):
            # Add the neural layer to the Message Passing neural network
            setattr(self, f"message_mlp_{i}", nn.Sequential(nn.Linear(dim * 2, dim),
                                                            nn.GELU(),
                                                            nn.Linear(dim, dim),
                                                            nn.GELU(),
                                                            )
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
            x = torch.cat([x, message], dim=1)
            x = layer(x)

        out = self.lin2(x)
        temperature = self.temperature(out)
        deformation = self.deformation(out)
        return torch.cat([temperature, deformation],1)

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
