from kioku.models import MLP
from kioku.schedulers import Scheduler, StaticScheduler
from kioku.utils import DEVICE
from kioku.loggers import Logger
import torch
from torch.distributions import Categorical
import numpy as np


class DiscreteActor:
    """An implementation of an actor for discrete action spaces.

    Attributes:
        model (MLP): The model used to predict the action probabilities.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] = [32, 32],
        create_optimizer: bool = True,
        learning_rate: Scheduler = StaticScheduler(3e-4, 0),
        gradient_clipping: float = 0.5,
    ) -> None:
        """
        Initialize the DiscreteActor.

        Args:
            observation_size (int): The size of the observation space.
            num_actions (int): The number of actions that can be taken.
            hidden_sizes (list[int]): The sizes of the hidden layers.
            create_optimizer (bool): Whether to create the optimizer.
            learning_rate (Scheduler): The learning rate scheduler.
            gradient_clipping (float): The maximum gradient norm for clipping.
        """

        self.model = MLP(observation_size, num_actions, hidden_sizes).to(DEVICE)
        if create_optimizer:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate.value(0))
            self._learning_rate = learning_rate
            self._gradient_clipping = gradient_clipping
        else:
            self._optimizer = None
            self._learning_rate = None
            self._gradient_clipping = None

    def act(self, observation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Select an action based on the observation.

        Args:
            observation (np.ndarray): The observation to predict the action probabilities for.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The action and the log probability of the action.
        """
        logits = self.model(observation)

        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        return action, action_log_prob

    def log_prob(self, observation: np.ndarray, action: np.ndarray) -> torch.Tensor:
        """Compute the log probability of taking a given action in a given state.
        
        Args:
            observation (np.ndarray): The observation to predict the action probabilities for.
            action (np.ndarray): The action to compute the log probability for.
        
        Returns:
            torch.Tensor: The log probability of taking the action.
        """
        logits = self.model(observation)

        action_dist = Categorical(logits=logits)
        action_log_prob = action_dist.log_prob(action)

        return action_log_prob

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """Optimize the actor model.

        Args:
            actor_loss (torch.Tensor): The loss of the actor model.
            step (int): The current step.
        """
        if self._optimizer is None:
            raise ValueError("The optimizer was not created.")

        self._learning_rate.adjust(self._optimizer, step)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self._gradient_clipping
        )
        self._optimizer.step()

        average_grad_norm = np.mean([torch.norm(param.grad).item() for param in self.model.parameters() if param.grad is not None])
        Logger.log_scalar("actor/learning_rate", self._learning_rate.value(step))
        Logger.log_scalar("actor/gradient_norm", average_grad_norm)

    def train(self) -> None:
        """Set the actor model to training mode."""
        self.model.train()
        if self._learning_rate:
            self._learning_rate.train()

    def test(self) -> None:
        """Set the actor model to evaluation mode."""
        self.model.eval()
        if self._learning_rate:
            self._learning_rate.test()
