from kioku.agents import Agent
from kioku.memory import NStepBuffer
from kioku.functions import DiscreteActor, Value
from kioku.vision import FeatureExtractor
from kioku.schedulers import Scheduler, StaticScheduler
from kioku.utils import Transition, DEVICE
from kioku.loggers import Logger
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class PPOAgent(Agent):
    """An implementation of the Proximal Policy Optimization (PPO) algorithm.

    See: https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        observation_size: int | tuple[int, int, int],
        num_actions: int,
        actor_hidden_sizes: list[int] = [32, 32],
        critic_hidden_sizes: list[int] = [32, 32],
        feature_extractor: FeatureExtractor | None = None,
        learning_rate: Scheduler = StaticScheduler(3e-4, 0),
        surrogate_clipping: float = 0.2,
        critic_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        normalize_advantages: bool = False,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        n_steps: int = 512,
        n_mini_batches: int = 4,
        n_training_steps: int = 4,
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initialize the PPO agent.

        Args:
            observation_size (int | tuple[int, int, int]): The size of the observation space.
                This can be either the size of the observation vector or the shape of the image.
            num_actions (int): The number of actions in the action space.
            actor_hidden_sizes (list[int]): The sizes of the hidden layers for the actor network.
            critic_hidden_sizes (list[int]): The sizes of the hidden layers for the critic network.
            feature_extractor (FeatureExtractor | None): The feature extractor for the agent.
                If None, the agent will use the observation directly.
            learning_rate (Scheduler): The learning rate of the shared optimizer.
            surrogate_clipping (float): The clipping value for policy ratios in the learning step.
            critic_coefficient (float): The coefficient for the critic loss.
            entropy_coefficient (float): The coefficient for the entropy term in the actor loss.
            normalize_advantages (bool): Whether to normalize the advantages.
            gamma (float): The discount factor for future rewards.
            lambda_ (float): The GAE lambda parameter.
            n_steps (int): The number of steps to use for n-step returns.
            n_mini_batches (int): The number of mini-batches to split a single learning batch into.
            n_training_steps (int): The number of times to iterate over a single learning batch.
            gradient_clipping (float): The max gradient norm for the agent's models.
        """
        if feature_extractor is not None:
            observation_size = feature_extractor.output_size
        self._feature_extractor = feature_extractor

        self._actor = DiscreteActor(
            observation_size=observation_size,
            num_actions=num_actions,
            hidden_sizes=actor_hidden_sizes,
            create_optimizer=False
        )
        self._critic = Value(
            input_size=observation_size,
            output_size=1,
            hidden_sizes=critic_hidden_sizes,
            create_optimizer=False
        )

        # Create a shared optimizer for the actor, critic, and feature extractor
        self._optimizer = torch.optim.Adam(
            self._total_parameters(),
            lr=learning_rate.value(0),
        )
        self._learning_rate = learning_rate
        self._gradient_clipping = gradient_clipping

        self._n_steps = n_steps
        self._n_step_buffer = NStepBuffer(self._n_steps)
        self._relevant_keys = [
            "observation",
            "action",
            "reward",
            "next_observation",
            "done",
            "action_log_prob",
        ]

        self._n_mini_batches = n_mini_batches
        self._n_training_steps = n_training_steps

        self._gamma = gamma
        self._lambda = lambda_
        self._surrogate_clipping = surrogate_clipping
        self._critic_coefficient = critic_coefficient
        self._entropy_coefficient = entropy_coefficient
        self._normalize_advantages = normalize_advantages

        self._step = 0

    def act(
        self, observation: np.ndarray, state: dict | None = None
    ) -> tuple[np.ndarray, dict | None]:
        """Choose an action based on the current observation.

        This function should be used when interacting with the environment.

        Args:
            observation (np.ndarray): The current observation.
            state (dict | None): The state of the agent.

        Returns:
            tuple[np.ndarray, dict | None]: The action to take, and the new state of the agent.
        """
        self._step += 1

        return self._choose_action(observation, state)

    def _choose_action(
        self, observation: np.ndarray, state: dict | None = None
    ) -> tuple[np.ndarray, dict | None]:
        """Choose an action based on the current observation.

        This function can be used when interacting with the environment or during an
        offline learning step.

        Args:
            observation (np.ndarray): The current observation.
            state (dict | None): The state of the agent.

        Returns:
            tuple[np.ndarray, dict | None]: The action to take, and the new state of the agent.
        """
        if self._feature_extractor is not None:
            observation = self._feature_extractor(observation)

        action, action_log_prob = self._actor.act(observation)
        action_log_prob = action_log_prob.unsqueeze(-1)
        action_entropy = -action_log_prob * torch.exp(action_log_prob)
        action = action.cpu().numpy()

        return action, { "action_log_prob": action_log_prob, "action_entropy": action_entropy }

    def _log_prob(
        self, observation: np.ndarray, action: np.ndarray, state: dict | None = None
    ) -> torch.Tensor:
        """Computes the log probability of taking a given action in a given state.

        The observation will not be processed by the feature extractor. This function assumes that
        this has already been done.

        Args:
            observation (np.ndarray): The current observation.
            action (np.ndarray): The action to compute the log probability of.
            state (dict | None): The state of the agent.

        Returns:
            torch.Tensor: The log probability of taking the action.
        """
        action_log_prob = self._actor.log_prob(observation, action)
        return action_log_prob

    def process_transition(self, transition: Transition) -> None:
        """Process a transition by storing it in the buffer.

        Args:
            transition (Transition): The transition to process.
        """
        transition.filter(self._relevant_keys)
        self._n_step_buffer.store(transition)

    def learn(self) -> None:
        """Train the agent for one step."""
        if not self._n_step_buffer.can_sample():
            return

        batch_transition = self._n_step_buffer.sample()
        observation, action, reward, next_observation, done, action_log_prob = (
            batch_transition[self._relevant_keys]
        )
        observation = torch.tensor(observation, dtype=torch.float32).to(DEVICE)
        action = torch.tensor(action, dtype=torch.float32).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).squeeze().to(DEVICE)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(DEVICE)
        done = torch.tensor(done, dtype=torch.bool).squeeze().to(DEVICE)
        action_log_prob = action_log_prob.squeeze()

        if self._feature_extractor is not None:
            observation_features = self._feature_extractor(observation)
            next_observation_features = self._feature_extractor(next_observation)

            value_estimate = self._critic.predict(observation_features).squeeze()
            next_value_estimate = self._critic.predict(next_observation_features).squeeze()
        else:
            value_estimate = self._critic.predict(observation).squeeze()
            next_value_estimate = self._critic.predict(next_observation).squeeze()

        advantage = self._compute_gae(
            reward, value_estimate.detach(), next_value_estimate.detach(), done
        )
        if self._normalize_advantages:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        returns = advantage + value_estimate.detach()

        total_actor_loss = 0
        total_entropy_loss = 0
        total_critic_loss = 0

        batch_size = self._n_steps // self._n_mini_batches

        dataset = TensorDataset(
            observation,
            action,
            advantage,
            returns,
            action_log_prob,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Run mini-batch training
        for _ in range(self._n_training_steps):
            for mini_batch in data_loader:
                (
                    observation,
                    action,
                    advantage,
                    returns,
                    action_log_prob,
                ) = mini_batch

                if self._feature_extractor is not None:
                    observation = self._feature_extractor(observation)

                new_action_log_prob = self._log_prob(observation, action)
                new_action_entropy = new_action_log_prob * -torch.exp(
                    new_action_log_prob
                )
                new_action_log_prob = new_action_log_prob.squeeze()

                ratio = torch.exp(new_action_log_prob - action_log_prob.detach())
                clipped_ratio = torch.clamp(
                    ratio, 1 - self._surrogate_clipping, 1 + self._surrogate_clipping
                )

                entropy_loss = self._entropy_coefficient * -new_action_entropy.mean()
                actor_loss = -torch.min(
                    ratio * advantage, clipped_ratio * advantage
                ).mean()
                actor_loss = actor_loss - entropy_loss

                value_estimate = self._critic.predict(observation).squeeze()
                critic_loss = self._critic_coefficient * F.mse_loss(
                    value_estimate, returns
                )

                total_loss = actor_loss + critic_loss
                self._optimize(total_loss)

                total_actor_loss += actor_loss
                total_entropy_loss += entropy_loss
                total_critic_loss += critic_loss

        total_learning_steps = self._n_training_steps * self._n_mini_batches
        avg_actor_loss = total_actor_loss / total_learning_steps
        avg_entropy_loss = total_entropy_loss / total_learning_steps
        avg_critic_loss = total_critic_loss / total_learning_steps

        Logger.log_scalar("ppo_agent/actor_loss", avg_actor_loss)
        Logger.log_scalar("ppo_agent/entropy_loss", avg_entropy_loss)
        Logger.log_scalar("ppo_agent/critic_loss", avg_critic_loss)
        Logger.log_scalar("ppo_agent/advantage/max", advantage.max())
        Logger.log_scalar("ppo_agent/advantage/min", advantage.min())
        Logger.log_scalar("ppo_agent/advantage/mean", advantage.mean())
        Logger.log_scalar("ppo_agent/value_estimate/max", value_estimate.max())
        Logger.log_scalar("ppo_agent/value_estimate/min", value_estimate.min())
        Logger.log_scalar("ppo_agent/value_estimate/mean", value_estimate.mean())
        Logger.log_scalar("ppo_agent/next_value_estimate/max", next_value_estimate.max())
        Logger.log_scalar("ppo_agent/next_value_estimate/min", next_value_estimate.min())
        Logger.log_scalar("ppo_agent/next_value_estimate/mean", next_value_estimate.mean())
        Logger.log_scalar("ppo_agent/returns/max", returns.max())
        Logger.log_scalar("ppo_agent/returns/min", returns.min())
        Logger.log_scalar("ppo_agent/returns/mean", returns.mean())

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor): The rewards for each step.
            values (torch.Tensor): Value estimates from the critic for each step.
            next_values (torch.Tensor): Next step value estimates from the critic.
            dones (torch.Tensor): Whether each step was terminal.

        Returns:
            torch.Tensor: The computed advantages.
        """
        td_errors = rewards + self._gamma * next_values * (~dones) - values
        advantage = 0
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        for t in reversed(range(self._n_steps)):
            advantage = self._gamma * self._lambda * advantage + td_errors[t]
            advantages[t] = advantage

        return advantages

    def _optimize(self, loss: torch.Tensor) -> None:
        """Optimize the actor, critic and feature extractor models.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
        """

        self._learning_rate.adjust(self._optimizer, self._step)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._total_parameters(), self._gradient_clipping)
        self._optimizer.step()

        average_grad_norm = np.mean([p.grad.norm(2).item() for p in self._total_parameters()])
        Logger.log_scalar("ppo_agent/learning_rate", self._learning_rate.value(self._step))
        Logger.log_scalar("ppo_agent/gradient", average_grad_norm)

    def _total_parameters(self) -> list:
        """Get a list of all the parameters in the agent's models.

        Returns:
            list[torch.Tensor]: A list of all the parameters in the agent's models.
        """
        actor_parameters = list(self._actor.model.parameters())
        critic_parameters = list(self._critic.model.parameters())
        if self._feature_extractor is not None:
            feature_parameters = list(self._feature_extractor.parameters())
        else:
            feature_parameters = []
        return actor_parameters + critic_parameters + feature_parameters

    def train(self) -> None:
        """Set the agent to training mode."""
        self._actor.train()
        self._critic.train()

    def test(self) -> None:
        """Set the agent to testing mode."""
        self._actor.test()
        self._critic.test()
