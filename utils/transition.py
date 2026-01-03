import torch
import numpy as np


class Transition:
    """Stores information like observations and actions gathered from experience.

    This class is used to store a transition, which includes observations, actions, and other related data
    collected during the interaction between the agent and environment.
    """

    def __init__(self, **kwargs: dict[str, np.ndarray | torch.Tensor]) -> None:
        """Initializes the transition object with the given data.

        Args:
            **kwargs (dict): The data to store in the transition object, where the keys are strings and
                             the values are either numpy arrays or torch tensors.

        Raises:
            TypeError: If any of the data is not a np.ndarray or torch.Tensor.
        """
        self._data: dict[str, np.ndarray | torch.Tensor] = {}

        for key, value in kwargs.items():
            if not isinstance(value, (np.ndarray, torch.Tensor)):
                raise TypeError(f"Expected np.ndarray or torch.Tensor for '{value}', got '{type(value)}' instead.")
            self._data[key] = value

    @classmethod
    def combine(cls, transitions: list["Transition"] | np.ndarray["Transition"]) -> "Transition":
        """Combines a batch of transitions into a single transition object.

        Args:
            transitions (list[Transition] | np.ndarray[Transition]): A batch of transitions to
                combine. Each transition in the batch must have the same keys, and have values
                of the same type (numpy arrays or torch tensors).

        Raises:
            ValueError: If the transitions list is empty.

        Returns:
            Transition: A new transition object with the combined data from the batch of transitions.
        """
        if len(transitions) == 0:
            raise ValueError("The transitions list is empty.")

        batch_data = {}
        for key in transitions[0].keys():
            if isinstance(transitions[0][key], torch.Tensor):
                batch_data[key] = torch.stack([t[key] for t in transitions], dim=0)
            else:
                # we can assume it's a numpy array because of the check in __init__
                batch_data[key] = np.stack([t[key] for t in transitions], axis=0)
        return Transition(**batch_data)

    def filter(self, keys: list[str]) -> "Transition":
        """Filters the transition object to include only the specified keys.

        Args:
            keys (list[str]): The keys to include in the filtered transition object.

        Returns:
            Transition: A new transition object with only the specified keys.

        Raises:
            KeyError: If any of the keys are not found in the transition object.
        """
        for key in keys:
            if key not in self._data:
                raise KeyError(f"Key '{key}' not found in transition data.")
        filtered_data = {key: self._data[key] for key in keys}
        return Transition(**filtered_data)

    def keys(self) -> list[str]:
        """Returns a list of all the keys in the transition object.

        Returns:
            list[str]: The keys in the transition object.
        """
        return list(self._data.keys())

    def __add__(self, other: "Transition") -> "Transition":
        """Adds the data of two transition objects.

        Args:
            other (Transition): The other transition object to add.

        Returns:
            Transition: A new transition object with the combined data.

        Raises:
            TypeError: If the other object is not a Transition.
            ValueError: If the keys of the two transitions do not match.
        """
        if not isinstance(other, Transition):
            raise TypeError(f"Expected Transition, got '{type(other)}' instead.")

        if self._data.keys() != other._data.keys():
            raise ValueError("Transition objects must have the same keys to be added.")

        combined_data = {}
        for key in self._data.keys():
            if isinstance(self._data[key], np.ndarray):
                combined_data[key] = np.concatenate((self._data[key], other._data[key]), axis=0)
            elif isinstance(self._data[key], torch.Tensor):
                combined_data[key] = torch.cat((self._data[key], other._data[key]), dim=0)

        return Transition(**combined_data)

    def __getitem__(self, keys: str | list[str]) -> np.ndarray | tuple[np.ndarray, ...]:
        """Retrieves the data stored in the transition object.

        If a single key is provided, the corresponding data is returned. If multiple keys are
        provided, a tuple of the corresponding data is returned.

        Args:
            keys (str | tuple[str, ...]): The key(s) to retrieve the data.

        Returns:
            np.ndarray | tuple[np.ndarray, ...]: The data corresponding to the key(s).

        Raises:
            KeyError: If the key(s) are not found in the transition object.
        """
        if isinstance(keys, list):
            return tuple(self[key] for key in keys)

        key = keys
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in transition data.")

        return self._data[key]

    def __repr__(self) -> str:
        """Returns a string representation of the transition object.

        Returns:
            str: The string representation of the transition object.
        """
        return f"Transition({self._data})"
