from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
from torch_geometric.data import Data


class SimpleMlp(MessagePassing):
    def __int__(self, in_channels: int, hidden_channels: int, out_channels: int, number_hidden: int):
        """ Simple MLP.

        This neural network is not using the graph and only send all nodes features to an MLP.

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
        """
        super().__init__(aggr="mean")

        self.name = "Simple MLP"

        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)

        self.network = MLP(in_channels=self.in_channels, hidden_channels=self.hidden, act="gelu",
                           out_channels=self.out_channels, num_layers=self.number_hidden)

    def forward(self, batch: Data) -> Tensor:
        # Check that all input features values are bellow 1
        assert (batch.x.max() <= 1)  # If assert is wrong: check data normalisation.

        return self.network(batch.x)
