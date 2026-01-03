from models import MLP
from schedulers import Scheduler, StaticScheduler
from utils import DEVICE
from loggers import Logger
import torch
from torch import optim


class Value:
    """A simple single-network value function approximator.

    This network is typically used for value functions in policy gradient methods like A2C.

    Attributes:
        model (MLP): The online network used for value estimation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [],
        create_optimizer: bool = True,
        learning_rate: Scheduler = StaticScheduler(3e-4),
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initializes the Value network.

        Args:
            input_size (int): The size of the input tensor.
            output_size (int): The size of the output tensor.
            hidden_sizes (list[int]): Sizes of hidden layers.
            create_optimizer (bool): Whether to create the optimizer.
            learning_rate (Scheduler): The learning rate of the optimizer.
            gradient_clipping (float): The maximum gradient norm for clipping.
        """
        self.model = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        if create_optimizer:
            self._optimizer = optim.Adam(self.model.parameters(), lr=learning_rate.value(0))
            self._learning_rate = learning_rate
            self._gradient_clipping = gradient_clipping
        else:
            self._optimizer = None
            self._learning_rate = None
            self._gradient_clipping = None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the value estimate for a given input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The predicted value estimate.
        """
        return self.model(x)

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """Optimizes the Value network.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
            step (int): The current training step.
        """
        if self._optimizer is None:
            raise ValueError("The optimizer was not created.")

        self._learning_rate.adjust(self._optimizer, step)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._gradient_clipping)
        self._optimizer.step()

        avg_gradient = sum(
            p.grad.abs().mean() for p in self.model.parameters()
        ) / len(list(self.model.parameters()))
        Logger.log_scalar("value/learning_rate", self._learning_rate.value(step))
        Logger.log_scalar("value/gradient", avg_gradient)

    def train(self) -> None:
        """Sets the Value network to training mode."""
        self.model.train()
        if self._learning_rate:
            self._learning_rate.train()

    def test(self) -> None:
        """Sets the Value network to evaluation mode."""
        self.model.eval()
        if self._learning_rate:
            self._learning_rate.test()
