# finetune_embedding/utils/seeding.py
import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across different components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: Enable deterministic algorithms, potentially impacting performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed} for Python random, NumPy, and PyTorch.")
