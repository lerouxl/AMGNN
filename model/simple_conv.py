import torch
from torch import Tensor
import torch_geometric as tg
from torch_geometric.data import Data
from model.operations import MLP


class SimpleSAGEConv(torch.nn.Module):
    """Simple deep learing model using multiple SAGE Conv layers.

    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, number_hidden: int, aggregator: str):
        """

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
        super(SimpleSAGEConv, self).__init__()

        self.name = "Simple Conv"

        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)
        self.out_channels = int(out_channels)
        self.in_channels = int(in_channels)

        self.encoder = MLP(in_channels=self.in_channels, hidden_channels=self.hidden, act="GELU",
                           out_channels=self.hidden, num_layers=2)
        self.decoder = MLP(in_channels=self.hidden, hidden_channels=self.hidden, act="GELU",
                           out_channels=self.out_channels, num_layers=2)

        conv_layers = []

        for _ in range(self.num_layers):
            conv_layers.append(tg.nn.conv.SAGEConv(in_channels=self.hidden, out_channels=self.hidden, ))
            conv_layers.append(torch.nn.GELU())

        self.conv_layers = torch.nn.Sequential(*conv_layers)

    def forward(self,  batch: Data) -> Tensor:
        """Apply the neural network to a data batch.

        From a graph, apply the simple graph convolution with an encoder and decoder.

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
        assert (batch.x.max() <= 1)

        x = self.encoder(batch.x)
        x = self.conv_layers(x, batch.edge_index)
        x = self.decoder(x)

        return x