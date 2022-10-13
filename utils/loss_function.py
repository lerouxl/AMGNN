import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing

def AMGNN_loss(batch, y, y_hat, lambda_weight=None):
    """
    Compute the loss of AMGNN. The loss function is defined as
    Loss = lambda_1 x L_mse + lambda_2 x L_gradient_deformation + lambda_3 x L_gradient_temperature

    With
        - L_mse: the MSE loss of y, y_hat.
        - L_gradient_deformation: the
    :return:
    """
    pass


def compute_gradient_loss(batch, y: Tensor, y_hat: Tensor) -> Tensor:
    """
    Compute the gradient edge loss on y and y_hat.
    Then the absolute of the average of the difference of the two gradient is done and returned as the gradient loss.

    :param batch: The batch data, containing the edge_index
    :param y: The label value of each nodes of the graph.
    :param y_hat: The predicted value of each nodes of the graph.
    :return: L_gradient: a tensor of shape [n_features] value of the loss.
    """
    # Load the gradient message parsing operator
    loss_function = gradient_loss_MP()

    loss = loss_function(y, y_hat, batch.edge_index)
    y_grad = loss[0]
    y_hat_grad = loss[1]

    # Do the mean
    L_gradient = torch.mean(y_grad - y_hat_grad)
    return L_gradient


class gradient_loss_MP(MessagePassing):
    """
    For each nodes of a graph, computes the average gradient of all connected edges.
    For this, a message passing technic is used, the values of each nodes are subtracted to there neighbors values.
    The absolute values of this subtraction is used to do the mean of all edges gradients values for nodes.

    """
    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, y: Tensor, y_hat: Tensor, edge_index: Tensor, return_gradient= False) -> Tensor:
        """
        Compute the gradient loss for a graph using the label and prediction.
        Can return the nodes gradients values if return_gradient is set to True
        :param y: The nodes values of shape [n_nodes].
        :param y_hat: The nodes predictions of shape [n_nodes].
        :param edge_index: The edge indices.
        :param return_gradient: bool, optional: if true, the gradient of each nodes is returned with the loss.
        :return: Tensor. Tensor of shape [2] with the mean of the absolute gradient of all features.
        :return: if return_gradient is True, grad, a tensor of shape [1, n_nodes] containing the average gradient values at each points.
        """
        # Merge the y and y_hat to create a tensor of shape [ n_nodes, dimension(y_i) + dimension(y_hat_i]
        x = torch.stack([y, y_hat]).t()
        # Compute the edges gradient.
        grad = self.propagate(edge_index, x=x).t()

        if not return_gradient:
            return torch.mean(grad, axis=1)
        else:
            return torch.mean(grad, axis=1), grad

    def message(self, x_i, x_j):
        """
        Do the absolute value and the subtraction between all nodes to compute the edges gradient.
        :param x_i: Nodes values
        :param x_j: Neighbours nodes values
        :return:
        """
        return torch.abs(x_i - x_j)