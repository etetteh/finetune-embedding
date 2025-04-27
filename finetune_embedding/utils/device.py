# finetune_embedding/utils/device.py
import logging

import torch

# Use absolute import
from finetune_embedding.config.settings import TrainingConfig

logger = logging.getLogger(__name__)


def determine_device() -> torch.device:
    """
    Determines the optimal computation device (CUDA, MPS, or CPU).
    """
    logger.info("Determining computation device...")
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA is available. Using device: {device}")
        try:
            logger.info(f"  Device Name: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            logger.info(f"  CUDA Capability: {cap[0]}.{cap[1]}")
        except Exception as e:
            logger.warning(f"Could not retrieve detailed CUDA device info: {e}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if MPS is built and available
        try:
            # Simple tensor op to verify MPS functionality
            torch.ones(1, device="mps")
            device = torch.device("mps")
            logger.info(
                "CUDA not available, but MPS (Apple Silicon GPU) is available and functional. Using device: mps"
            )
        except Exception as e:
            logger.warning(f"MPS available but failed test ({e}). Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA and MPS are not available. Using CPU. Training will be significantly slower."
        )

    logger.info(f"Selected device: {device}")
    return device


def update_precision_flags(
    training_config: TrainingConfig, device: torch.device
) -> None:
    """
    Adjusts FP16/BF16 flags in the TrainingConfig based on device capabilities
    and user preferences. Modifies the config object in place.
    """
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    # MPS BF16 support is often problematic, treat as unsupported for now
    # bf16_supported = bf16_supported or (device.type == 'mps')

    if training_config.use_fp16 and training_config.use_bf16:
        logger.warning(
            "Both use_fp16 and use_bf16 requested. Prioritizing BF16 if supported, otherwise FP16."
        )
        if bf16_supported:
            training_config.use_fp16 = False  # Prioritize BF16
            logger.info("BF16 supported and prioritized. Disabling FP16 flag.")
        else:
            training_config.use_bf16 = (
                False  # BF16 not supported, keep FP16 (will be checked next)
            )
            logger.info(
                "BF16 not supported by device. Keeping FP16 flag (will be checked)."
            )

    # Check FP16 support (primarily CUDA)
    if training_config.use_fp16 and device.type != "cuda":
        logger.warning(f"FP16 requested but device is {device.type}. Disabling FP16.")
        training_config.use_fp16 = False

    # Check BF16 support again after potential priority adjustment
    if training_config.use_bf16 and not bf16_supported:
        logger.warning(
            f"BF16 requested but not supported by the device ({device.type}). Disabling BF16."
        )
        training_config.use_bf16 = False

    logger.info(
        f"Final precision flags: use_fp16={training_config.use_fp16}, use_bf16={training_config.use_bf16}"
    )
