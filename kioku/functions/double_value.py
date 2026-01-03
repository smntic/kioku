from kioku.models import MLP
from kioku.schedulers import Scheduler, StaticScheduler
from kioku.utils import DEVICE
from kioku.loggers import Logger
import torch
from torch import optim


class DoubleValue:
    """A simple Double Value network.

    This network could be used for the Q and V functions, or any other value function.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [],
        learning_rate: Scheduler = StaticScheduler(3e-4),
        transition_rate: float = 0.005,
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initializes the Double Value network.

        Args:
            input_size (int): The size of the input tensor.
            output_size (int): The size of the output tensor.
            learning_rate (Scheduler): The learning rate of the optimizer.
            transition_rate (float): The rate at which the target network transitions to the online
                network.
        """
        self._online_network = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        self._target_network = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        self._copy_online_to_target()

        self._optimizer = optim.Adam(self._online_network.parameters(), lr=learning_rate.value(0))
        self._learning_rate = learning_rate
        self._transition_rate = transition_rate
        self._gradient_clipping = gradient_clipping

    def predict(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Defines the value prediction of the Double Value network.

        Args:
            x (torch.Tensor): The input tensor.
            target (bool): Whether to use the target network or the online network.
        Returns:
            torch.Tensor: The output tensor after passing through the Double Value network.
        """
        if target:
            with torch.no_grad():
                return self._target_network(x)
        return self._online_network(x)

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """Optimize the Double Value network.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
            step (int): The current step.
        """
        self._learning_rate.adjust(self._optimizer, step)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online_network.parameters(), max_norm=self._gradient_clipping)
        self._optimizer.step()
        self._update_target_network()

        avg_gradient = 0
        for param in self._online_network.parameters():
            avg_gradient += param.grad.abs().mean()
        avg_gradient /= len(list(self._online_network.parameters()))
        Logger.log_scalar("double_value/learning_rate", self._learning_rate.value(step))
        Logger.log_scalar("double_value/gradient", avg_gradient)

    def _update_target_network(self) -> None:
        """Updates the target network with the online network's parameters."""
        target_state_dict = self._target_network.state_dict()
        online_state_dict = self._online_network.state_dict()
        for key in target_state_dict:
            target_state_dict[key] = self._transition_rate*online_state_dict[key] + (1-self._transition_rate)*target_state_dict[key]
        self._target_network.load_state_dict(target_state_dict)

    def _copy_online_to_target(self) -> None:
        """Copies the online network's parameters to the target network."""
        self._target_network.load_state_dict(self._online_network.state_dict())

    def train(self) -> None:
        """Sets the Double Value network to training mode."""
        self._online_network.train()
        self._target_network.train()
        self._learning_rate.train()

    def test(self) -> None:
        """Sets the Double Value network to evaluation mode."""
        self._online_network.eval()
        self._target_network.eval()
        self._learning_rate.test()
