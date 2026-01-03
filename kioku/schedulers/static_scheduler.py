from kioku.schedulers import Scheduler


class StaticScheduler(Scheduler):
    """A scheduler that always returns the same value."""

    def __init__(self, value: float, test_value: float | None = None) -> None:
        """Initialize the static scheduler.

        Args:
            value (float): The value to return.
            test_value (float | None): The value to return in test mode.
        """
        self._value = value
        self._test_mode = False
        self._test_value = test_value if test_value is not None else self._value

    def value(self, step: int) -> float:
        """Get the value of the scheduler at a given step.

        Args:
            step (int): The current step.

        Returns:
            float: The value of the scheduler at the given step.
        """
        if self._test_mode:
            return self._test_value
        return self._value

    def train(self) -> None:
        """Set the scheduler to training mode."""
        self._test_mode = False

    def test(self) -> None:
        """Set the scheduler to testing mode."""
        self._test_mode = True
