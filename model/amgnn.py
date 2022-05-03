from torch import optim, nn
from typing import Tuple
import torch
import wandb
#from model.GNCConv import GCNConv
from torch_geometric.nn import MessagePassing, GCNConv
from dataloader.arc_dataset import ARCDataset
import pytorch_lightning as pl


class AMGNNmodel(pl.LightningModule):
    """Main class of AMGNN. This Pytorch lightning class will deal with the model loading, training, testing, validation
     and logings"""

    def __init__(self, config):
        super().__init__()
        self.configuration = config
        self.network = NeuralNetwork(self.configuration.input_channels, self.configuration.hidden_channels,
                                     self.configuration.out_channels)
        self.Floss = nn.functional.mse_loss
        self.lr_val = self.configuration.learning_rate
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        #self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        """Train loop of the neural network"""
        _, loss, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("train loss", loss)
        self.log("train displacement loss", loss_disp)
        self.log("train temperature loss", loss_temp)
        return loss

    def test_step(self, batch, batch_idx):
        """The test set is NOT used during training, it is ONLY used once the model has been trained to see how the
         model will do in the real-world."""
        _, loss, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("test loss", loss)
        self.log("test displacement loss", loss_disp)
        self.log("test temperature loss", loss_temp)

    def validation_step(self, batch, batch_idx):
        """During training, it’s common practice to use a small portion of the train split to determine when the model
         has finished training.As a rule of thumb, we use 20% of the training set as the validation set.
         This number varies from dataset to dataset."""
        _, loss, loss_disp, loss_temp = self._get_preds_loss(batch)
        self.log("val loss", loss)
        self.log("val displacement loss", loss_disp)
        self.log("val temperature loss", loss_temp)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr_val)
        return optimizer

    def _get_preds_loss(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """The train, test and validation function are the same so their are regrouped here.
        Return:
            y_hat: the prediction of the neural network,
            loss: the actual loss of the neural network"""
        y_hat = self.network(batch)
        temp_hat = y_hat[:,0]
        temp = batch.y[:,0]

        disp_hat = y_hat[:,1:-1]
        disp = batch.y[:,1:-1]
        loss_temp = self.Floss(temp_hat, temp)
        loss_disp = self.Floss(disp_hat, disp)
        loss = loss_disp + loss_temp #self.Floss(y_hat, batch.y)
        return y_hat, loss, loss_disp, loss_temp


class NeuralNetwork(MessagePassing):
    def __init__(self,in_channels,  hidden_channels, out_channels):
        super().__init__(aggr='add')

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act = nn.LeakyReLU()

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        x = self.lin1(x)
        x = self.act(x)

        x = self.conv1(x=x, edge_index=edge_index.long(), edge_weight=edge_attr)
        x = self.act(x)

        x = self.conv2(x=x, edge_index=edge_index.long(), edge_weight=edge_attr)
        x = self.act(x)

        x= self.lin2(x)
        return x
