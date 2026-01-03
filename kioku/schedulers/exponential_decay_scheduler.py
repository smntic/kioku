from kioku.schedulers import Scheduler


class ExponentialDecayScheduler(Scheduler):
    """An exponential decay scheduler."""

    def __init__(
        self, begin: float, end: float, time: float, test_value: float | None = None
    ) -> None:
        """Initialize the exponential decay scheduler.

        Args:
            begin (float): The initial value of the decay.
            end (float): The final value of the decay.
            time (float): The time over which the decay occurs.
            test_value (float | None): The value to return in test mode.
        """
        self._begin = begin
        self._end = end
        self._time = time
        self._test_mode = False
        self._test_value = self._end if test_value is None else test_value

    def value(self, step: int) -> float:
        """Get the value of the decay at a given step.

        Args:
            step (int): The current step.

        Returns:
            float: The value of the decay at the given step.
        """
        if self._test_mode:
            return self._test_value
        if step >= self._time:
            return self._end
        return self._begin * (self._end / self._begin) ** (step / self._time)

    def train(self) -> None:
        """Set the scheduler to training mode."""
        self._test_mode = False

    def test(self) -> None:
        """Set the scheduler to testing mode."""
        self._test_mode = True
