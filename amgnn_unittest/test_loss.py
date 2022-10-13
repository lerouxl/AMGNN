"""Test loss.py"""

import unittest
import torch
from torch import Tensor
from utils import loss_function
from torch_geometric.data import Data
import torch_geometric as tg

tg.set_debug(True)


def simple_graph():
    """Create a simple graph, with 3 points.
    Return the adjacency matrix and label.

    A----B----C
    With:
        A=1
        B=2
        C=1
    :return:
        data : torch geo data object of the graph
    """
    # Define the edge index
    edge_index = torch.tensor([[0, 1],
                               [1, 2]], dtype=torch.long)
    edge_index = tg.utils.to_undirected(edge_index)

    # Define the label
    y = torch.tensor([1.0, 2.1, 3.2])

    # Define the position
    pos = torch.tensor([[1, 0, 0],
                        [2, 0, 0],
                        [3, 0, 0]])

    data = Data(x=y, y=y, edge_index=edge_index, pos=pos)
    return data


def show_grap(data: Data):
    """Display a torch geometric data as a graph
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    nx.draw(tg.utils.to_networkx(data))
    plt.show()


class Test_loss(unittest.TestCase):
    """
    Check that the loss computation is well made.
    """
    def test_grad_zeros(self):
        """
        Check that the gradient loss is zeros with y a zeros tensor and y_hat a ones tensors.

        Graph:
            A========B========C
                Y  Y_hat
            A:  0  1
            B:  0  1
            C:  0  1

        Expect that the mean gradient of Y and Y_hat is 0.
        :return:
        """
        # Create data
        data = simple_graph()
        # show_grap(data)
        y = torch.zeros_like(data.y)
        y_hat = torch.ones_like(data.y)

        # Check gradient computation
        loss_f = loss_function.gradient_loss_MP()

        loss = loss_f(y, y_hat, data.edge_index)
        y_grad = loss[0]
        y_hat_grad = loss[1]

        self.assertEqual(torch.tensor([0]), y_grad,
                         "The gradient of an graph with all nodes values equal to 0 must be 0")
        self.assertEqual(torch.tensor([0]), y_hat_grad,
                         "The gradient of an graph with all nodes values equal to 1 must be 0")

        # Check gradient loss computation
        grad_loss = loss_function.compute_gradient_loss(data, y, y_hat)
        self.assertEqual(torch.tensor([0]), grad_loss,
                         "The mean loss must be 0")


    def test_grad_ones(self):
        """
        Check that the gradient loss is zeros with y a zeros tensor and y_hat a ones tensors.

        Graph:
            A========B========C
                Y  Y_hat
            A:  1  -1
            B:  2  -2
            C:  3  -3

        Expect that the mean gradient of Y and Y_hat is 1.
        :return:
        """
        data = simple_graph()
        # show_grap(data)
        y = torch.arange(data.y.shape[0], dtype=torch.float) + 1.0
        y_hat = - torch.arange(data.y.shape[0], dtype=torch.float) - 1.0

        loss_f = loss_function.gradient_loss_MP()

        loss = loss_f(y, y_hat, data.edge_index)
        y_grad = loss[0]
        y_hat_grad = loss[1]

        self.assertEqual(torch.tensor([1]), y_grad,
                         "The gradient of an graph with all nodes values equal to 0 must be 0")
        self.assertEqual(torch.tensor([1]), y_hat_grad,
                         "The gradient of an graph with all nodes values equal to 1 must be 0")

        # Check gradient loss computation
        grad_loss = loss_function.compute_gradient_loss(data, y, y_hat)
        self.assertEqual(torch.tensor([0]), grad_loss,
                         "The mean loss must be 0")

    def test_loss_zeros(self):
        """Test the case when the loss is equal to zeros:
        - Y = Y_hat
        - labmda_weight = 0
        """
        # Generate data
        data = simple_graph()
        y = torch.ones((3,4), dtype=torch.float)
        y_hat = torch.zeros((3,4), dtype=torch.float)

        # Loss(y,y) = 0
        zero = torch.tensor([0])
        loss = loss_function.AMGNN_loss(data, y, y)
        self.assertEqual(zero, loss, "The loss of y - y should be zeros")

        # Lambda weight set to 0
        lambda_weight = torch.zeros((3,1))
        loss = loss_function.AMGNN_loss(data, y, y_hat, lambda_weight)
        self.assertEqual(zero, loss, "The loss with lambda zeros should be zeros")

    def test_loss_no_grad(self):
        """Test remove the grad calculation with lambda_weight and test the MSE loss.

        Graph:
            A========B========C
        With Y = 0,sqrt(1),sqrt(2),sqrt(3) and Y_hat = 0.
        The MSE values must be 1.5 (18 / 12).

        """
        # Generate data
        data = simple_graph()
        y = torch.unsqueeze(torch.arange(4, dtype=torch.float),0) # [1,4]
        y = torch.sqrt(y)
        y = y.expand(3,4) # [3,4]

        y_hat = torch.zeros((3,4), dtype=torch.float)

        # Loss(y,y) = 1.5
        target = torch.tensor([1.5])

        # Focus on the MSE loss
        lambda_weight = torch.zeros((3,1))
        lambda_weight[0][0] = 1
        loss = loss_function.AMGNN_loss(data, y, y_hat, lambda_weight)
        self.assertEqual(target, loss, "This MSE loss must be equal to 1.5")




