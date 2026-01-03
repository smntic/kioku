from models import MLP
from schedulers import Scheduler, StaticScheduler
from utils import DEVICE
from loggers import Logger
import torch
from torch import optim


class MinDoubleValue:
    """A class that creates two networks and takes the minimum of their outputs for a value
        function.

    It manages the optimization of both networks and their target network transitions.

    See: https://arxiv.org/pdf/1509.06461
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
        """Initializes the MinDoubleValue network with two networks (online and target for each).

        Args:
            input_size (int): The size of the input tensor.
            output_size (int): The size of the output tensor.
            hidden_sizes (list[int]): The sizes of the hidden layers.
            learning_rate (Scheduler): The learning rate scheduler.
            transition_rate (float): The rate at which target networks transition.
            gradient_clipping (float): The maximum gradient norm for clipping.
        """
        self._online_network_1 = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        self._online_network_2 = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        self._target_network_1 = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        self._target_network_2 = MLP(input_size, output_size, hidden_sizes).to(DEVICE)
        self._copy_online_to_target()

        self._optimizer = optim.Adam(
            list(self._online_network_1.parameters()) + list(self._online_network_2.parameters()),
            lr=learning_rate.value(0),
        )

        self._learning_rate = learning_rate
        self._transition_rate = transition_rate
        self._gradient_clipping = gradient_clipping

    def predict(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Predicts the minimum value from both networks.

        Args:
            x (torch.Tensor): The input tensor.
            target (bool): Whether to use the target networks.

        Returns:
            torch.Tensor: The minimum of the two outputs.
        """
        if target:
            with torch.no_grad():
                pred_1 = self._target_network_1(x)
                pred_2 = self._target_network_2(x)
                return torch.min(pred_1, pred_2)
        else:
            pred_1 = self._online_network_1(x)
            pred_2 = self._online_network_2(x)
            return torch.min(pred_1, pred_2)

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """Optimizes both networks with respect to the given loss.

        Args:
            loss (torch.Tensor): The loss to backpropagate.
            step (int): The current optimization step.
        """
        self._learning_rate.adjust(self._optimizer, step)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._online_network_1.parameters())
            + list(self._online_network_2.parameters()),
            max_norm=self._gradient_clipping,
        )
        self._optimizer.step()
        self._update_target_network()

        avg_gradient_1 = 0
        for param in self._online_network_1.parameters():
            avg_gradient_1 += param.grad.abs().mean()
        avg_gradient_1 /= len(list(self._online_network_1.parameters()))

        avg_gradient_2 = 0
        for param in self._online_network_2.parameters():
            avg_gradient_2 += param.grad.abs().mean()
        avg_gradient_2 /= len(list(self._online_network_2.parameters()))

        Logger.log_scalar("min_double_value/gradient_1", avg_gradient_1)
        Logger.log_scalar("min_double_value/learning_rate", self._learning_rate.value(step))
        Logger.log_scalar("min_double_value/gradient_2", avg_gradient_2)

    def _update_target_network(self) -> None:
        """Updates the target networks with a weighted average of online networks' parameters."""
        self._update_target_network_params(self._target_network_1, self._online_network_1)
        self._update_target_network_params(self._target_network_2, self._online_network_2)

    def _update_target_network_params(
        self, target_network: MLP, online_network: MLP
    ) -> None:
        """Updates the target network with a weighted average of online network's parameters.

        Args:
            target_network (MLP): The target network to update.
            online_network (MLP): The online network to transition from.
        """
        target_state_dict = target_network.state_dict()
        online_state_dict = online_network.state_dict()
        for key in target_state_dict:
            target_state_dict[key] = self._transition_rate*online_state_dict[key] + (1-self._transition_rate)*target_state_dict[key]
        target_network.load_state_dict(target_state_dict)

    def _copy_online_to_target(self) -> None:
        """Copies the online networks' parameters to the target networks."""
        self._target_network_1.load_state_dict(self._online_network_1.state_dict())
        self._target_network_2.load_state_dict(self._online_network_2.state_dict())

    def train(self) -> None:
        """Sets both online and target networks to training mode."""
        self._online_network_1.train()
        self._online_network_2.train()
        self._target_network_1.train()
        self._target_network_2.train()
        self._learning_rate.train()

    def test(self) -> None:
        """Sets both online and target networks to evaluation mode."""
        self._online_network_1.eval()
        self._online_network_2.eval()
        self._target_network_1.eval()
        self._target_network_2.eval()
        self._learning_rate.test()
