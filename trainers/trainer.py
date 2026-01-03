from agents import Agent
from environments import Environment
from utils import Transition
from loggers import Logger
from typing import Any


class Trainer:
    """The Trainer class trains an agent on a given environment."""

    def __init__(self, agent: Agent, environment: Environment) -> None:
        """Initializes the Trainer class.

        Args:
            agent (Agent): The agent to train.
            environment (Environment): The environment to train the agent on.
        """
        self._agent = agent
        self._environment = environment

    def train(self, num_steps: int) -> None:
        """Trains the agent for a specified number of steps.

        Args:
            num_steps (int): The number of steps to train the agent for.
        """
        self._agent.train()

        episode_complete = False
        episode_reward = 0
        for step in range(1, num_steps + 1):
            if episode_complete or step == 1:
                observation = self._environment.reset()
                agent_state = {}

            if episode_complete:
                Logger.log_scalar("train/episode_reward", episode_reward)

                episode_complete = False
                episode_reward = 0

            action, agent_state = self._agent.act(observation, agent_state)
            next_observation, reward, done, truncated = self._environment.step(action)
            transition = Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                **agent_state
            )
            self._agent.process_transition(transition)

            observation = next_observation

            self._agent.learn()

            if done.any() or truncated.any():
                episode_complete = True

            Logger.log_scalar("train/reward", reward)
            Logger.log_scalar("train/action", action)

            episode_reward += reward

    def test(self, num_episodes: int) -> dict[str, Any]:
        """Tests the agent for a specified number of episodes and returns the results.

        Returns:
            dict[str, Any]: The results dictionary containing information about the testing.
        """
        self._agent.test()

        episode_rewards = []

        for _ in range(num_episodes):
            observation = self._environment.reset()
            agent_state = {}

            episode_complete = False
            episode_reward = 0

            while not episode_complete:
                action, agent_state = self._agent.act(observation, agent_state)
                next_observation, reward, done, truncated = self._environment.step(action)

                observation = next_observation

                episode_reward += reward

                if done.any() or truncated.any():
                    episode_complete = True

                Logger.log_scalar("test/reward", reward)
                Logger.log_scalar("test/action", action)

            episode_rewards.append(episode_reward)
            Logger.log_scalar("test/episode_reward", episode_reward)

        avg_reward = sum(episode_rewards) / num_episodes
        return {"avg_reward": avg_reward, "episode_rewards": episode_rewards}
