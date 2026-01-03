from agents import Agent
from memory import ExperienceReplayBuffer
from functions import DoubleValue
from schedulers import Scheduler, ExponentialDecayScheduler, StaticScheduler
from utils import Transition, DEVICE
from loggers import Logger
import torch
import torch.nn.functional as F
import numpy as np


class DQNAgent(Agent):
    """A Deep Q-Network (DQN) agent.

    See: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] = [32, 32],
        learning_rate: Scheduler = StaticScheduler(1e-3, 0),
        epsilon: Scheduler = ExponentialDecayScheduler(1, 0.01, 3000, 0),
        gamma: float = 0.99,
        transition_rate: float = 0.005,
        memory_size: int = 5000,
        batch_size: int = 32,
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initialize the DQN agent.

        Args:
            observation_size (int): The size of the observation.
            num_actions (int): The number of actions the agent can take.
            hidden_sizes (list[int]): The sizes of the models' hidden layers.
            learning_rate (Scheduler): The learning rate.
            epsilon (Scheduler): The epsilon value.
            gamma (float): The gamma value.
            transition_rate (float): The transition rate.
            memory_size (int): The size of the memory.
            batch_size (int): The size of the batch.
            gradient_clipping (float): The max gradient norm for the agent's models.
        """
        self._epsilon = epsilon
        self._gamma = gamma

        self._model = DoubleValue(
            input_size=observation_size,
            output_size=num_actions,
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            transition_rate=transition_rate,
            gradient_clipping=gradient_clipping,
        )
        self._num_actions = num_actions

        self._batch_size = batch_size
        self._memory = ExperienceReplayBuffer(memory_size, batch_size)
        self._relevant_keys = [
            "observation",
            "action",
            "reward",
            "next_observation",
            "done",
        ]

        self._step = 0

    def act(self, observation: np.ndarray, state: dict = None) -> tuple[np.ndarray, dict | None]:
        """Choose an action based on the current observation.

        Args:
            observation (np.ndarray): The current observation.
            state (dict | None): The state of the agent.

        Returns:
            tuple[np.ndarray, dict | None]: The action to take, and the new state.
        """
        self._step += 1

        chosen_action = None
        epsilon_value = self._epsilon.value(self._step)

        if np.random.rand() < epsilon_value:
            chosen_action = np.random.randint(self._num_actions)
        else:
            # Select the action with the highest predicted q value
            with torch.no_grad():
                q_values = self._model.predict(observation)
                q_values = q_values.cpu().numpy()
            chosen_action = np.argmax(q_values)

        chosen_action = np.array([chosen_action])

        Logger.log_scalar("dqn_agent/epsilon", epsilon_value)

        return chosen_action, state

    def process_transition(self, transition: Transition) -> None:
        """Process a transition by either storing it in the memory buffer or learning on-policy.

        Args:
            transition (Transition): The transition to process.
        """
        transition = transition.filter(self._relevant_keys)
        self._memory.store(transition)

    def learn(self) -> None:
        """Train the agent on a batch of experiences."""
        if not self._memory.can_sample():
            return

        batch_transition = self._memory.sample()
        observation, action, reward, next_observation, done = batch_transition[
            self._relevant_keys
        ]

        observation = torch.tensor(observation, dtype=torch.float32).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(DEVICE)
        done = torch.tensor(done, dtype=torch.float32).to(DEVICE)

        target_q_values = (
            self._model.predict(next_observation, target=True).max(dim=1).values
        )
        target_q_values = reward.squeeze(dim=1) + self._gamma*target_q_values*(1-done.squeeze(dim=1))

        current_q_values = self._model.predict(observation)
        current_q_values = current_q_values.gather(1, action).squeeze(dim=1)

        q_loss = F.mse_loss(current_q_values, target_q_values)

        self._model.optimize(q_loss, self._step)

        Logger.log_scalar("dqn_agent/loss", q_loss.item())
        Logger.log_scalar("dqn_agent/pred_q_value/max", current_q_values.max().item())
        Logger.log_scalar("dqn_agent/pred_q_value/min", current_q_values.min().item())
        Logger.log_scalar("dqn_agent/pred_q_value/mean", current_q_values.mean().item())
        Logger.log_scalar("dqn_agent/target_q_value/max", target_q_values.max().item())
        Logger.log_scalar("dqn_agent/target_q_value/min", target_q_values.min().item())
        Logger.log_scalar("dqn_agent/target_q_value/mean", target_q_values.mean().item())

    def train(self) -> None:
        """Set the agent to training mode."""
        self._model.train()
        self._epsilon.train()

    def test(self) -> None:
        """Set the agent to testing mode."""
        self._model.test()
        self._epsilon.test()
