import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import LSTMAggregation, MeanAggregation

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
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, act: str):
        super(Custom_MLP, self).__init__()
        test = [nn.Linear(in_channels, hidden_channels), nn.GELU()]
        for i in range(num_layers):
            test.extend([nn.Linear(hidden_channels, hidden_channels), nn.GELU()])
        test.extend([nn.Linear(hidden_channels, out_channels)])

        self.layers = nn.Sequential(*test)

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