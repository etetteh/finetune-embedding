# finetune_embedding/model/lora.py
import logging
from typing import Optional

# Import necessary types and exceptions
from peft import LoraConfig, TaskType, PeftModel # Ensure PeftModel is imported
from sentence_transformers import SentenceTransformer
from ..config.settings import LoRAConfig # Assuming LoRAConfig is here
from ..exceptions import ConfigurationError, ModelError # Import exceptions

logger = logging.getLogger(__name__)

def add_lora_adapter(model: SentenceTransformer, lora_config: LoRAConfig) -> None:
    """Adds a LoRA adapter to the model if configured."""
    if not lora_config.use_lora:
        logger.info("Skipping LoRA adapter addition (use_lora=False).")
        return

    if not model:
         raise ModelError("Model must be initialized before adding LoRA.")

    logger.info("Adding LoRA adapter...")

    # --- REMOVE REDUNDANT CHECKS - Pydantic handles these ---
    # if lora_config.rank <= 0: raise ConfigurationError(f"LoRA rank must be > 0, got {lora_config.rank}")
    # if lora_config.alpha <= 0: raise ConfigurationError(f"LoRA alpha must be > 0, got {lora_config.alpha}")
    # Pydantic also handles dropout validation (0.0 <= d <= 1.0)

    try:
        # Import peft types locally within the try block to handle ImportError correctly
        from peft import LoraConfig, TaskType, PeftModel

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            inference_mode=False,
        )
        logger.debug(f"LoRA Config: {peft_config}")

        try:
             # Access the underlying transformer model correctly
             base_transformer_model = model._first_module().auto_model
             if isinstance(base_transformer_model, PeftModel):
                 logger.warning("Base model already has PEFT config. Adding another LoRA adapter. Ensure this is intended.")
        except Exception:
             # Log if the check fails, but don't stop the process
             logger.debug("Could not check if base model is PeftModel.")

        model.add_adapter(peft_config)
        logger.info(f"LoRA adapter added successfully with config: {peft_config}")

    except ImportError:
        logger.error("PEFT library not found. Please install it: pip install peft", exc_info=True)
        # Raise ConfigurationError as it's a setup/dependency issue
        raise ConfigurationError("PEFT library is required for LoRA.")
    except Exception as e:
        logger.error(f"Unexpected error adding LoRA adapter: {e}", exc_info=True)
        # Raise ModelError for runtime issues during adapter addition
        raise ModelError(f"Failed to add LoRA adapter: {e}") from e

