from torch import optim, nn
from typing import Tuple
import torch
import wandb
#from model.GNCConv import GCNConv
from torch_geometric.nn import MessagePassing, GCNConv
from dataloader.arc_dataset import ARCDataset
from torch_geometric.data import Data
import pytorch_lightning as pl
import os.path as osp
from utils.loss_function import AMGNN_loss
from model.transfomer_utils import PositionalEncoding
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric as tg

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
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        """Train loop of the neural network"""
        _, loss, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train loss gradient displacement", loss_disp,batch_size=self.batch_size)
        self.log("train loss gradient temperature", loss_temp, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """The test set is NOT used during training, it is ONLY used once the model has been trained to see how the
         model will do in the real-world."""
        y_hat, loss, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("test loss", loss, batch_size=self.batch_size)
        self.log("test loss gradient displacement", loss_disp, batch_size=self.batch_size)
        self.log("test loss gradient temperature", loss_temp, batch_size=self.batch_size)

        # Save the output
        batch_cpu = batch.to("cpu")
        batch_cpu.y = torch.hstack([batch_cpu.y, y_hat.to("cpu")])
        torch.save(batch_cpu, osp.join(osp.join(self.configuration["raw_data"], "test_output"), f'{batch_idx}.pt'))


    def validation_step(self, batch, batch_idx):
        """During training, it’s common practice to use a small portion of the train split to determine when the model
         has finished training.As a rule of thumb, we use 20% of the training set as the validation set.
         This number varies from dataset to dataset."""
        _, loss, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("val loss", loss, batch_size=self.batch_size)
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

        loss, loss_disp, loss_temp = AMGNN_loss(batch, batch.y, y_hat, detail_loss=True)

        return y_hat, loss, loss_disp, loss_temp


class NeuralNetwork(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, aggregator, number_hidden):
        super().__init__(aggr=aggregator) # Test multi agreagation with cat to extract the neighboor
        self.hidden = int(hidden_channels)
        self.number_hidden = int(number_hidden)

        self.lin = nn.Linear(in_channels, self.hidden)
        self.lin2 = nn.Linear(self.hidden, out_channels)

        dim = self.hidden
        for i in range(self.number_hidden):
            # Add the neural layer to the Message Passing neural network
            setattr(self, f"message_mlp_{i}", nn.Sequential(nn.Linear(dim*2, dim),
                                                            nn.GELU(),
                                                            nn.Linear(dim,dim),
                                                            nn.GELU(),
                                                            )
                    )

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 4-5: Start propagating messages.
        x_ = [x]
        for i in range(self.number_hidden):
            # Found the Nth layer of the neural network
            layer = getattr(self, f"message_mlp_{i}")
            # Append its results in x_
            x_.append(self.propagate(edge_index, x=x_[-1], model= layer ))


        # Add all layers outputs together for the skip connections
        x = torch.sum(
            torch.stack(x_), axis=0
        )
        out = self.lin2(x)
        return out

    def message(self, x_i, x_j, model):
        """

        :param x_i: has shape [E, out_channels]
        :param x_j: has shape [E, out_channels]
        :param model: The model to apply to [x_i , x_j - x_i]
        :return:
        """

        # This is an edges convolution
        # https://arxiv.org/abs/1801.07829

        msg = torch.cat([x_i, x_j - x_i], dim=1)
        msg = model(msg)

        return msg

class Aggregation(torch.nn.Module):
    r"""An abstract base class for implementing custom aggregations.

    Aggregation can be either performed via an :obj:`index` vector, which
    defines the mapping from input elements to their location in the output:

    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Notably, :obj:`index` does not have to be sorted:

    .. code-block::

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Assign each element to one of three sets:
       index = torch.tensor([0, 0, 1, 0, 2, 0, 2, 1, 0, 2])

       output = aggr(x, index)  #  Output shape: [4, 64]

    Alternatively, aggregation can be achieved via a "compressed" index vector
    called :obj:`ptr`. Here, elements within the same set need to be grouped
    together in the input, and :obj:`ptr` defines their boundaries:

    .. code-block::

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Define the boundary indices for three sets:
       ptr = torch.tensor([0, 4, 7, 10])

       output = aggr(x, ptr=ptr)  #  Output shape: [4, 64]

    Note that at least one of :obj:`index` or :obj:`ptr` must be defined.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or edge features
          :math:`(|\mathcal{E}|, F_{in})`,
          index vector :math:`(|\mathcal{V}|)` or :math:`(|\mathcal{E}|)`,
        - **output:** graph features :math:`(|\mathcal{G}|, F_{out})` or node
          features :math:`(|\mathcal{V}|, F_{out})`
    """

    def forward(self, x, index, ptr, dim_size,dim = -2) :
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            index (torch.LongTensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.LongTensor, optional): If given, computes the
                aggregation based on sorted inputs in CSR representation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
        """
        pass