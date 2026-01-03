from utils import DEVICE
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any


class FeatureExtractor(nn.Module):
    """Used to extract a feature vector from an image.

    Attributes:
        output_size (int): The size of the output feature vector.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int,
        conv_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        pool_kernel_sizes: list[int],
        pool_strides: list[int],
    ) -> None:
        """The constructor for the FeatureExtractor class.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input image (height, width, channels).
            output_size (int): The desired size of the output feature vector.
            conv_channels (list[int]): List of number of channels for each convolutional layer.
            kernel_sizes (list[int]): List of kernel sizes for each convolutional layer.
            strides (list[int]): List of strides for each convolutional layer.
            paddings (list[int]): List of padding values for each convolutional layer.
            pool_kernel_sizes (list[int]): List of kernel sizes for pooling layers.
            pool_strides (list[int]): List of strides for pooling layers.
        """
        super(FeatureExtractor, self).__init__()

        self.output_size = output_size
        self._input_channels, self._input_height, self._input_width = input_shape

        self._conv_layers = nn.ModuleList()
        self._pool_layers = nn.ModuleList()
        in_channels = self._input_channels
        for out_channels, kernel_size, stride, pad, pool_kernel_size, pool_stride in zip(conv_channels, kernel_sizes, strides, paddings, pool_kernel_sizes, pool_strides):
            self._conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
            self._pool_layers.append(nn.MaxPool2d(pool_kernel_size, pool_stride))
            in_channels = out_channels

        self._conv_output_height = self._input_height
        self._conv_output_width = self._input_width
        for stride, kernel_size, pad, pool_stride, pool_kernel_size in zip(strides, kernel_sizes, paddings, pool_strides, pool_kernel_sizes):
            self._conv_output_height = (self._conv_output_height + 2 * pad - kernel_size) // stride + 1
            self._conv_output_width = (self._conv_output_width + 2 * pad - kernel_size) // stride + 1
            self._conv_output_height = (self._conv_output_height - pool_kernel_size) // pool_stride + 1
            self._conv_output_width = (self._conv_output_width - pool_kernel_size) // pool_stride + 1

        self._fc = nn.Linear(in_channels * self._conv_output_height * self._conv_output_width, output_size)

    def forward(self, x: Any) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (Any): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output feature vector of shape (batch_size, output_size).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        added_batch_dim = False
        if len(x.shape) == 3:
            added_batch_dim = True
            x = x.unsqueeze(0)

        for conv_layer, pool_layer in zip(self._conv_layers, self._pool_layers):
            x = F.relu(conv_layer(x))
            x = pool_layer(x)

        x = x.reshape(x.size(0), -1)
        x = self._fc(x)
        if added_batch_dim:
            x = x.squeeze(0)

        return x
