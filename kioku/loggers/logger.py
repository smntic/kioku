from torch.utils.tensorboard import SummaryWriter
import os
import time


class Logger:
    """Singleton class for logging scalar values to TensorBoard."""

    _logger_instance = None
    _step_dict = {}

    @classmethod
    def _get_logger_instance(cls, log_dir=".logs/kioku"):
        """Returns the instance of the logger. 
        
        If the instance does not exist, it creates a new instance.

        Args:
            log_dir (str): The directory to save the logs.

        Returns:
            SummaryWriter: The instance of the logger.
        """
        if cls._logger_instance is None:
            # Create a unique log directory for each run based on the current time
            unique_log_dir = os.path.join(log_dir, time.strftime("%m-%d_%H:%M:%S"))
            cls._logger_instance = SummaryWriter(unique_log_dir)
        return cls._logger_instance

    @classmethod
    def log_scalar(cls, tag, value):
        """Logs a scalar value to TensorBoard.

        Args:
            tag (str): The tag to assign to the scalar value.
            value (float): The scalar value to log.
        """
        logger = cls._get_logger_instance()

        # Increment the step for the given tag
        if tag not in cls._step_dict:
            cls._step_dict[tag] = 0
        cls._step_dict[tag] += 1

        logger.add_scalar(tag, value, cls._step_dict[tag])

    @classmethod
    def close(cls):
        """Closes the logger."""
        if cls._logger_instance:
            cls._logger_instance.close()
            cls._logger_instance = None
