from torch import Tensor
from torch_geometric.nn import MessagePassing
from model.operations import MLP
from torch_geometric.data import Data


class SimpleGnn(MessagePassing):
    """"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, number_hidden: int, aggregator: str):
        """ Simple Graph neural network.

        This neural network is using a encoder decoder architecture with an massage passing them inbetween.

        Parameters
        ----------
        in_channels: int
            Number of features per nodes.
        hidden_channels: int
            Size of the hidden layer of the MLP.
        out_channels: int
            Number of output features per nodes.
        number_hidden: int
            Number of hidden layer of the MLP
        aggregator: str
            Type of aggregation function to use for passing message in the graph.
        """
        super().__init__(aggr=str(aggregator))

        self.name = "Simple GNN"

        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)

        self.encoder = MLP(in_channels=self.in_channels, hidden_channels=self.hidden, act="GELU",
                           out_channels=self.hidden, num_layers=self.number_hidden)
        self.decoder = MLP(in_channels=self.hidden, hidden_channels=self.hidden, act="GELU",
                           out_channels=self.out_channels, num_layers=self.number_hidden)

    def forward(self, batch: Data) -> Tensor:
        """Apply the neural network to a data batch.

        From a graph, apply the default message passing network with an encoder and decoder.
        Parameters
        ----------
        batch: Data
            Torch geometric graph.

        Returns
        -------
        torch.Tensor:
            Node predictions.
        """
        # Check that all input features values are bellow 1
        assert (batch.x.max() <= 1)  # If assert is wrong: check data normalisation.

        x = self.encoder(batch.x)

        x = self.propagate(batch.edge_index, x=x)
        x = self.decoder(x)
        return x

    def message(self, x_j):
        """Compute the message passing by the graph.
        The message is the features of the neighbors node.
        """
        return x_j