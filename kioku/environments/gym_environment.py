from kioku.environments import Environment
import gymnasium as gym
import numpy as np


class GymEnvironment(Environment):
    """Wrapper for the gymnasium environment.

    Attributes:
        action_size (int): The number of actions that can be taken.
        observation_size (int | tuple[int, int, int]): The dimension of the observation space.
        continuous (bool): Whether the environment has a continuous action space.
    """

    def __init__(self, environment_name: str, render_mode: str | None = None) -> None:
        """Initializes the given environment

        Args:
            environment_name (str): The name of the environment.
            render_mode (str): The mode for rendering the environment.
        """
        self._environment = gym.make(environment_name, render_mode=render_mode)

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        random_seed = np.random.randint(0, 2**32)
        observation, _ = self._environment.reset(seed=random_seed)
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Takes a step in the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The next observation (np.ndarray),
                - The reward (np.ndarray),
                - Whether the episode is done (np.ndarray),
                - Whether the episode was truncated (np.ndarray).
        """
        action = action.item()
        observation, reward, done, truncated, _ = self._environment.step(action)
        reward = np.array([reward], dtype=np.float32)
        done = np.array([done], dtype=bool)
        truncated = np.array([truncated], dtype=bool)
        return observation, reward, done, truncated

    @property
    def action_size(self) -> int:
        return (
            self._environment.action_space.n
            if not self.continuous
            else self._environment.action_space.shape[0]
        )

    @property
    def observation_size(self) -> int | tuple[int, int, int]:
        return self._environment.observation_space.shape[0]

    @property
    def continuous(self) -> bool:
        return isinstance(self._environment.action_space, gym.spaces.Box)
