from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    """Abstract base class for an environment.

    Attributes:
        action_size (int): The number of actions that can be taken.
        observation_size (int | tuple[int, int, int]): The dimension of the observation space.
        continuous (bool): Whether the environment has a continuous action space.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The next observation (np.ndarray),
                - The reward (np.ndarray),
                - Whether the episode is done (np.ndarray),
                - Whether the episode was truncated (np.ndarray).
        """
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """For discrete environments: The number of actions that can be taken.
        For continuous environments: The dimension of the action space.
        """
        pass

    @property
    @abstractmethod
    def observation_size(self) -> int | tuple[int, int, int]:
        """The dimension of the observation space."""
        pass

    @property
    @abstractmethod
    def continuous(self) -> bool:
        """Whether the environment has a continuous action space."""
        pass
