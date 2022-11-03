import torch
from torch.nn.functional import mse_loss
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


def AMGNN_loss(batch: Data, y: Tensor, y_hat: Tensor, lambda_weight: Tensor = None, detail_loss: bool = False):
    """Compute the loss of AMGNN.
    The loss function is defined as
     \[ Loss = \lambda_1 \cdot Loss_{mse} + \lambda_2 \cdot Loss_{gradient deformation} + \lambda_3 \cdot Loss_{gradient temperature}\]
    With:
        - **Loss<sub>mse</sub>**: the MSE loss of *y*, *y_hat*.
        - **Loss<sub>gradient deformation</sub>**: the mean of the gradients losses X, Y and Z gradient.
        - **Loss<sub>gradient temperature</sub>**: the gradient loss of the temperature.
        - **lambda<sub>i</sub>**: the ith weight in *lambda_weight*. If *lambda_weigth* is None, the all lambda<sub>i</sub> are set to 1.

    :param: batch: The batch data containing the edge_index.
    :param: y: The label tensor of shape [n_nodes, 4] (temperature, X disp,Y disp,Z disp).
    :param: y_hat: The prediction tensor of shape [n_nodes, 4] (temperature, X disp,Y disp,Z disp).
    :param: lambda_weight: The weight of the weighted sum of the loss (shape (3,1))
    :param: detail_loss: boolean, if false, only return the loss,
                        if true, return: loss, l_gradient_deformation, l_gradient_temperature
    :return: Tensor the loss value (shape [1]).
    """

    # If lambda_weight is None, then create a tensor of 1.
    if lambda_weight is None:
        lambda_weight = torch.ones(3, device=y.device).view(3,1) # [3,1]
    else:
        lambda_weight = lambda_weight.to(y.device) # Check that the lambda vector is on the good device.

    # Compute the mse loss
    l_mse = mse_loss(y_hat[:, 0], y[:, 0]) + 10*mse_loss(y_hat[:, 1:], y[:, 1:]) # [1]

    # Compute the temperature gradient loss
    l_gradient_temperature = compute_gradient_loss(batch=batch, y=y[:, 0], y_hat=y_hat[:, 0]) # [1]

    # Compute the X gradient loss
    l_gradient_x = compute_gradient_loss(batch=batch, y=y[:, 1], y_hat=y_hat[:, 1]) # [1]

    # Compute the Y gradient loss
    l_gradient_y = compute_gradient_loss(batch=batch, y=y[:, 2], y_hat=y_hat[:, 2]) # [1]

    # Compute the Z gradient loss
    l_gradient_z = compute_gradient_loss(batch=batch, y=y[:, 3], y_hat=y_hat[:, 3]) # [1]

    # Compute l_gradient_deformation
    l_gradient_deformation = torch.mean(torch.stack([l_gradient_x,
                                                    l_gradient_y,
                                                    l_gradient_z])) # [1]

    # Compute the loss by adding the sub losses and weight them
    loss = torch.stack([l_mse,
                       l_gradient_temperature,
                       l_gradient_deformation]).view(1, 3) # [1,3]
    # weighted sum
    loss = torch.mm(loss, lambda_weight) # [1,1]
    loss = torch.squeeze(loss,0) # [1]

    if detail_loss:
        return loss, l_mse, l_gradient_deformation, l_gradient_temperature
    else:
        return loss


def compute_gradient_loss(batch, y: Tensor, y_hat: Tensor) -> Tensor:
    """
    Compute the gradient edge loss on y and y_hat.
    Then the absolute of the average of the difference of the two gradient is done and returned as the gradient loss.

    :param batch: The batch data, containing the edge_index
    :param y: The label value of each nodes of the graph.
    :param y_hat: The predicted value of each nodes of the graph.
    :return: L_gradient: a tensor of shape [1] value of the loss.
    """
    # Load the gradient message parsing operator
    loss_function = gradient_loss_MP()

    loss = loss_function(y, y_hat, batch.edge_index)
    y_grad = loss[0]
    y_hat_grad = loss[1]

    # Do the mean
    L_gradient = torch.mean(torch.abs(y_grad - y_hat_grad))
    return L_gradient


class gradient_loss_MP(MessagePassing):
    """
    For each nodes of a graph, computes the average gradient of all connected edges.
    For this, a message passing technic is used, the values of each nodes are subtracted to there neighbors values.
    The absolute values of this subtraction is used to do the mean of all edges gradients values for nodes.

    """

    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, y: Tensor, y_hat: Tensor, edge_index: Tensor, return_gradient=False) -> Tensor:
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
