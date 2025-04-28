# finetune_embedding/model/loaders.py
import logging
from pathlib import Path

import torch
from peft import PeftModel  # Assuming PeftModel might be used
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

# Assuming these are defined elsewhere
from ..config.settings import ModelConfig, TrainingConfig
from ..exceptions import ModelError

logger = logging.getLogger(__name__)


def initialize_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,  # Needed for precision flags
    device: torch.device,
) -> SentenceTransformer:
    """Initializes the SentenceTransformer model based on configuration."""
    logger.info(f"Initializing model: {model_config.model_name_or_path}")

    model_kwargs = {}
    target_dtype = None
    # Determine dtype based on effective precision flags (assuming they are set correctly)
    if training_config.use_bf16:
        target_dtype = torch.bfloat16
        logger.info("Setting model load dtype to torch.bfloat16")
    elif training_config.use_fp16:
        target_dtype = torch.float16
        logger.info("Setting model load dtype to torch.float16")

    if target_dtype:
        model_kwargs["torch_dtype"] = target_dtype

    model_card_data = SentenceTransformerModelCardData(
        language=model_config.language,
        license=model_config.license_type,
        model_name=model_config.model_name_or_path.split("/")[-1],
    )

    try:
        cache_folder_str = (
            str(model_config.cache_dir)
            if isinstance(model_config.cache_dir, Path)
            else model_config.cache_dir
        )
        model = SentenceTransformer(
            model_name_or_path=model_config.model_name_or_path,
            model_card_data=model_card_data,
            trust_remote_code=model_config.trust_remote_code,
            cache_folder=cache_folder_str,
            model_kwargs=model_kwargs,
            device=str(device),
        )

        # Check if it's already a PEFT model (best effort)
        try:
            if isinstance(model._first_module().auto_model, PeftModel):
                logger.warning(
                    f"Model loaded from '{model_config.model_name_or_path}' seems to already be a PEFT model. Ensure this is intended."
                )
        except Exception:
            logger.debug("Could not check if loaded model is PeftModel.")

        logger.info(
            f"Model '{model_config.model_name_or_path}' initialized successfully on device '{model.device}'."
        )
        # Log parameter dtype if possible
        try:
            param_dtype = next(model.parameters()).dtype
            logger.info(f"Data type of first model parameter after load: {param_dtype}")
        except Exception:
            pass  # Ignore if model has no parameters or inspection fails

        return model

    except OSError as e:  # Catch specific file/repo not found errors
        logger.error(
            f"Failed to load model '{model_config.model_name_or_path}'. Check path/name and network. Error: {e}",
            exc_info=True,
        )
        raise ModelError(f"Failed to load model: {e}") from e
    # --- FIX: Add generic exception handler ---
    except Exception as e:
        logger.error(
            f"Unexpected error initializing model '{model_config.model_name_or_path}': {e}",
            exc_info=True,
        )
        raise ModelError(f"Model initialization failed: {e}") from e
    # --- END FIX ---
