from abc import ABC, abstractmethod
import torch


class Scheduler(ABC):
    """An abstract class for a scheduler."""

    @abstractmethod
    def value(self, step: int) -> float:
        """Get the value of the scheduler at a given step.

        Args:
            step (int): The current step.

        Returns:
            float: The value of the scheduler at the given step.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """Set the scheduler to training mode."""
        pass

    @abstractmethod
    def test(self) -> None:
        """Set the scheduler to testing mode."""
        pass

    def adjust(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Adjust the learning rate of the optimizer using the scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to adjust.
            step (int): The current step.
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.value(step)
