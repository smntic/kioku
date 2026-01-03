import numpy as np
import torch


def set_seed(seed: int):
    """Sets the seed globally.

    Args:
        seed (int): The seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
