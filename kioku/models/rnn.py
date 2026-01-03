from kioku.utils import DEVICE
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any


class RNN(nn.Module):
    """A Recurrent Neural Network (RNN) model.

    Attributes:
        _layers (nn.ModuleList): List of linear layers that make up the fully-connected section.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        inner_state_size: int = 64,
        rnn_layers: int = 1,
        hidden_sizes: list[int] = [],
    ) -> None:
        """Initializes the RNN model with a variable number of layers.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            inner_state_size (int): The size of the inner state of the RNN.
            hidden_sizes (list[int]): The sizes of the hidden layers.
        """
        super().__init__()

        fully_connected_layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.rnn = nn.RNN(
            input_size=fully_connected_layer_sizes[-1],
            hidden_size=inner_state_size,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.fully_connected_layers = nn.ModuleList()
        for i in range(len(fully_connected_layer_sizes) - 1):
            self.fully_connected_layers.append(nn.Linear(fully_connected_layer_sizes[i], fully_connected_layer_sizes[i + 1]))

    def forward(
        self, x: Any, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Defines the forward pass of the RNN.

        Args:
            x (Any): The input tensor.
            state (torch.Tensor): The hidden state of the RNN.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensor after passing through the MLP, and
                the hidden state of the RNN.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(DEVICE)

        if state is None:
            state = torch.zeros(1, x.size(0), self.rnn.hidden_size)
        for layer in self.fully_connected_layers[:-1]:
            x = F.relu(layer(x))
        x = self.fully_connected_layers[-1](x)

        return x
