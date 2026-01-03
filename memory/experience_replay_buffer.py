from utils import Transition
from loggers import Logger
import numpy as np


class ExperienceReplayBuffer:
    """A class used to represent an Experience Replay Buffer."""

    def __init__(self, max_size: int = 1000000, batch_size: int = 32) -> None:
        """Initializes the Experience Replay Buffer.

        Args:
            max_size (int): The maximum size of the buffer.
            batch_size (int): The size of the batch to sample.
        """
        self._max_size = max_size
        self._full = False
        self._data_index = 0

        self._batch_size = batch_size

        self._data = np.empty(max_size, dtype=Transition)

    def store(self, transition: Transition) -> None:
        """Stores a transition in the buffer.

        Args:
            transition (Transition): The transition to store.
        """
        self._data[self._data_index] = transition
        self._advance_index()

        Logger.log_scalar("experience_replay_buffer/buffer_size", len(self))

    def sample(self) -> Transition:
        """Samples a batch of transitions from the buffer.

        Raises:
            ValueError: If the buffer does not have enough transitions to sample a full batch.

        Returns:
            Transition: A batch of transitions with combined data.
        """
        if not self.can_sample():
            raise ValueError("Not enough transitions in the buffer to sample a full batch.")

        indices = np.random.choice(len(self), self._batch_size, replace=False)
        chosen_transitions = self._data[indices]
        batch_transition = Transition.combine(chosen_transitions)
        return batch_transition

    def can_sample(self) -> bool:
        """Checks if the buffer has enough transitions to sample a full batch.

        Returns:
            bool: Whether the buffer has enough transitions to sample a full batch.
        """
        return len(self) >= self._batch_size

    def _advance_index(self) -> None:
        """Advance the data index pointer by one.

        This is a circular buffer. If the pointer exceeds the maximum index, reset it to 0 and set the buffer to full.
        """
        self._data_index += 1
        if self._data_index >= self._max_size:
            self._data_index = 0
            self._full = True

    def __len__(self) -> int:
        """Return the current length of the buffer."""
        if self._full:
            return self._max_size
        return self._data_index
