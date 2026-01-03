import numpy as np
from kioku.environments import Environment


class DoorEnvironment(Environment):
    """A reinforcement learning environment where an agent must choose the correct door after a
    series of steps.

    The environment provides observations and rewards, and handles episode termination. The agent
    must choose the correct door after the correct door is flashed for a given number of steps.

    Attributes:
        action_size (int): The number of actions available to the agent.
        observation_size (int | tuple[int, int, int]): The dimensionality of the observation space.
        continuous (bool): Whether the action space is continuous. Always False for this
            environment.
    """

    def __init__(
        self,
        num_doors: int = 10,
        flash_steps: int = 10,
        reward_delay: int = 10,
        print_observation: bool = False,
    ) -> None:
        """Initializes the environment with the given parameters.

        Args:
            num_doors (int): The number of doors in the environment.
            flash_steps (int): The number of steps the correct door will be flashed.
            reward_delay (int): The delay (in steps) before the agent can receive a reward for
                choosing the correct door.
            print_observation (bool): If True, the observation will be printed at each step.
        """
        self._num_doors = num_doors
        self._flash_steps = flash_steps
        self._reward_delay = reward_delay
        self._print_observation = print_observation
        self._max_steps = 30

    def reset(self) -> np.ndarray:
        """Resets the environment to the initial state and returns the initial observation.

        The correct door is chosen randomly, and the observation vector is initialized with a 1 at
            the correct door.

        Returns:
            np.ndarray: The initial observation (one-hot encoded vector with a 1 at the correct
                door).
        """
        self._chosen_door = np.random.randint(0, self._num_doors)
        observation = np.zeros(self._num_doors, dtype=np.float32)
        observation[self._chosen_door] = 1

        if self._print_observation:
            print(f"observation: {observation}")

        self._step = 0
        return observation

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Executes one step in the environment based on the given action.

        Args:
            action (np.ndarray): The action taken by the agent (index of the chosen door).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The next observation (np.ndarray),
                - The reward (np.ndarray),
                - Whether the episode is done (np.ndarray),
                - Whether the episode was truncated (np.ndarray).
        """
        self._step += 1

        observation = np.zeros(self._num_doors, dtype=np.float32)
        if self._step < self._flash_steps:
            observation[self._chosen_door] = 1

        if self._print_observation:
            print(f"observation: {observation}, action: {action}")

        reward = np.array([0], dtype=np.float32)
        chosen_door = action.item()
        if chosen_door != 0:
            if (
                chosen_door == self._chosen_door + 1
                and self._step >= self._flash_steps + self._reward_delay
            ):
                reward += 1
            else:
                reward -= 1

        done = np.array([0], dtype=bool)
        truncated = np.array([int(self._step >= self._max_steps)], dtype=bool)

        return observation, reward, done, truncated

    @property
    def action_size(self) -> int:
        """Returns the number of actions available to the agent.

        Returns:
            int: The number of actions (number of doors + 1 for the "no action" case).
        """
        return self._num_doors + 1

    @property
    def observation_size(self) -> int | tuple[int, int, int]:
        """Returns the dimensionality of the observation space.

        Returns:
            int: The size of the observation space (number of doors).
        """
        return self._num_doors

    @property
    def continuous(self) -> bool:
        """Returns whether the action space is continuous.

        Returns:
            bool: Always False, as the action space is discrete in this environment.
        """
        return False
