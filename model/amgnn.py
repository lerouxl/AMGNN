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
from torch_geometric.nn.aggr import LSTMAggregation, MeanAggregation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from utils.logs import log_point_cloud_to_wandb
import numpy as np


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
        self.network = NeuralNetwork(self.configuration["input_channels"], self.configuration["hidden_channels"],
                                     self.configuration["out_channels"], self.configuration['aggregator'],
                                     self.configuration["number_hidden_layers"])
        self.lr = float(self.configuration["learning_rate"])
        self.batch_size = int(config["batch_size"])
        self.lambda_weight = torch.tensor(config["lambda_parameters"], dtype=torch.float32).view(3, 1)
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.save_hyperparameters()

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
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train loss mse", loss_mse, batch_size=self.batch_size)
        self.log("train loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("train loss gradient temperature", loss_temp, batch_size=self.batch_size)
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
        self.log("val loss", loss, batch_size=self.batch_size)
        self.log("val loss mse", loss_mse, batch_size=self.batch_size)
        self.log("val loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("val loss gradient temperature", loss_temp, batch_size=self.batch_size)

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
        scheduler = ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "train loss"
                }

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
        # Output of the network
        y_hat = self.network(batch)

        # Compute the losses
        loss, loss_mse, loss_disp, loss_temp = AMGNN_loss(batch, batch.y, y_hat, detail_loss=True,
                                                          lambda_weight=self.lambda_weight)

        return y_hat, loss, loss_mse, loss_disp, loss_temp


class TestNeuralNetwork(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, aggregator: str, number_hidden: int):
        # Simple MLP for test purpose
        super().__init__(aggr="mean")
        print("THIS IS A TEST NEURAL NETWORK")

        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)

        self.test_MLP = MLP(in_channels=self.in_channels, hidden_channels=self.hidden, act="gelu",
                            out_channels=self.out_channels, num_layers=10, dropout=0.3, )

    def forward(self, batch: Data) -> Tensor:
        assert (batch.x.max() <= 1)
        return self.test_MLP(batch.x)


class Custom_MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, act:str):
        super(Custom_MLP, self).__init__()
        test = [nn.Linear(in_channels, hidden_channels), nn.GELU()]
        for i in range(num_layers):
            test.extend([nn.Linear(hidden_channels, hidden_channels), nn.GELU()])
        test.extend([nn.Linear(hidden_channels, out_channels)])

        self.layers =nn.Sequential(*test)

    def forward(self, x):
        return self.layers(x)


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
        # super().__init__(aggr=LSTMAggregation(in_channels=hidden_channels, out_channels=hidden_channels))
        super().__init__(aggr="mean")

        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)

        self.relu_act = torch.nn.ReLU()

        # Input layer
        self.encoder = Custom_MLP(in_channels=15, hidden_channels=self.hidden, act="gelu",
                           out_channels=self.hidden, num_layers=3)

        # The message_mlp will take as input two (decoded nodes ||  past_features || normalised_coordinated)
        # and output just a self.hidden dimension
        self.message_mlp = Custom_MLP(in_channels=(self.hidden + 7) * 2 + 1, hidden_channels=self.hidden, act="gelu",
                               out_channels=self.hidden + 7, num_layers=3, )

        # Output layer
        self.decoder_temp = Custom_MLP(in_channels=self.hidden + 7, hidden_channels=self.hidden, act="gelu",
                           out_channels=1, num_layers=3, )
        self.decoder_displacement = Custom_MLP(in_channels=self.hidden + 7, hidden_channels=self.hidden, act="gelu",
                           out_channels=3, num_layers=3, )

        dim = self.hidden
        for i in range(self.number_hidden):
            # Add the neural layer to the Message Passing neural network
            setattr(self, f"message_mlp_{i}", nn.Sequential(nn.Linear(dim * 2, dim),
                                                            nn.GELU(),
                                                            nn.Linear(dim, dim)
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
        edge_index = batch.edge_index

        edges_features = batch.edge_attr  # Extract the unnormalized edge size  [n_edge, 1]
        past_features = batch.x[:, 8:12]  # Extract the past temperature, and past displacement [n_nodes, 4]
        normalised_coordinated = batch.x[:, 19:]  # Extract the node position [n_nodes, 3]
        x = torch.hstack([batch.x[:, :8], batch.x[:, 12:19]])  # [n_nodes, 15]

        # Encode the printing parameters, nodes types and printing features
        x = self.encoder(x)  # [n_nodes, self.hidden]
        x = torch.hstack([x, past_features, normalised_coordinated])  # [n_nodes, self.hidden + 7]

        for i in range(self.number_hidden):
            # Found the Nth layer of the neural network
            layer = getattr(self, f"message_mlp_{i}")
            # Append its results in x_
            message = self.propagate(edge_index, x=x, edge_attr=edges_features)
            # x = torch.cat([x, message], dim=1)
            # x = layer(x)
            x = message

        prediction = torch.hstack([self.relu_act(self.decoder_temp(x)),
                                   self.decoder_displacement(x)])
        return prediction

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr) -> Tensor:
        """ Compute the message of each edges.

        The message of each edges is calculated as the concatenation of
        \[ \\frac{x_i}{edge_ij} || \\frac{x_j}{edge_ij}\]
        \[ x_i || x_j || e_{ij} \]

        .. todo:: select the used equation

        Parameters
        ----------
        x_i : Tensor
            Features of the receiver node, of shape [Edge number, out_channels].
        x_j : Tensor
            Features of the neighbour node, of shape [Edge number, out_channels].
        edge_attr : Tensor
            Edge feature (edge size).

        Returns
        -------
        Tensor
            Output of the neural network block.
        """

        msg = torch.hstack([x_i, x_j, edge_attr])
        msg = self.message_mlp(msg)

        return msg
