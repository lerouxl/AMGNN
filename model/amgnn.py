import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import wandb
from model.GNCConv import GCNConv

class model():
    """Main class of AMGNN. This class will deal with the model loading, training, testing, validation and logings"""

    def __init__(self):
        self.configuration = wandb.config
        self.network = NeuralNetwork(self.configuration.hidden_channels, self.configuration.out_channels)

    def build(self):
        """Create the neural network using wandb parameters"""
        pass

    def train(self):
        """Train the neural network"""
        pass

    def validate(self):
        """Test the neural network on a test or validation dataset"""
        pass

    def predict(self):
        """Apply the neural network to make prediction """
        pass

    def load_data(self):
        """Load and preprocess the dataset"""
        pass


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
