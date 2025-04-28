# finetune_embedding/training/losses.py
import logging
from typing import Any, Callable, Dict

import torch.nn as nn  # Import torch.nn
from sentence_transformers import SentenceTransformer, losses

# Use absolute imports
from finetune_embedding.exceptions import ConfigurationError, ModelError

logger = logging.getLogger(__name__)

# Type alias for loss creation functions - Update return type
LossCreator = Callable[..., nn.Module]


def _create_mnrl_loss(
    model: SentenceTransformer, **kwargs
) -> losses.MultipleNegativesRankingLoss:
    """Creates MultipleNegativesRankingLoss."""
    logger.info("Selected MultipleNegativesRankingLoss.")
    return losses.MultipleNegativesRankingLoss(model)


def _create_cosent_loss(model: SentenceTransformer, **kwargs) -> losses.CoSENTLoss:
    """Creates CoSENTLoss."""
    logger.info("Selected CoSENTLoss.")
    return losses.CoSENTLoss(model)


def _create_softmax_loss(
    model: SentenceTransformer, num_labels: int, **kwargs
) -> losses.SoftmaxLoss:
    """Creates SoftmaxLoss, validating num_labels and embedding dimension."""
    if num_labels <= 1:
        raise ConfigurationError("SoftmaxLoss requires num_labels > 1.")

    emb_dim = model.get_sentence_embedding_dimension()
    if not emb_dim:
        raise ModelError("Could not determine embedding dimension for SoftmaxLoss.")

    logger.info(
        f"Selected SoftmaxLoss with num_labels={num_labels}, embedding_dim={emb_dim}."
    )
    return losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=emb_dim,
        num_labels=num_labels,
    )


# --- Dispatch Dictionary ---
LOSS_CREATORS: Dict[str, LossCreator] = {
    "triplet": _create_mnrl_loss,  # Typically MNRL is used for triplets in SBERT examples
    "pair-score": _create_cosent_loss,
    "pair-class": _create_softmax_loss,
    # Add other formats and their corresponding loss creators here
}


# --- Refactored Main Function ---
def create_loss_function(
    model: SentenceTransformer,
    effective_format: str,
    num_labels: int,  # Required for SoftmaxLoss
) -> nn.Module:  # Add return type hint nn.Module
    """Creates the appropriate loss function based on the effective dataset format using dispatch."""
    logger.info(f"Creating loss function for effective format '{effective_format}'.")
    if not model:
        raise ModelError("Model must be initialized before creating loss.")

    try:
        # --- Dispatch to Specific Creator ---
        creator_func = LOSS_CREATORS.get(effective_format)
        if not creator_func:
            raise ConfigurationError(
                f"Unsupported effective format '{effective_format}' for loss creation."
            )

        # Prepare arguments for the creator function
        # Pass only relevant arguments to avoid unexpected keyword errors
        creator_args: Dict[str, Any] = {"model": model}
        if effective_format == "pair-class":
            creator_args["num_labels"] = num_labels

        # Call the selected creator function
        loss: nn.Module = creator_func(
            **creator_args
        )  # Ensure loss is typed as nn.Module

        logger.info(f"Loss function created: {type(loss).__name__}")
        return loss

    # --- Exception Handling (Order Matters) ---
    except (
        ConfigurationError
    ) as ce:  # Catch specific config errors raised intentionally
        logger.error(
            f"Configuration error creating loss function: {ce}", exc_info=False
        )
        raise  # Re-raise the original ConfigurationError

    except ModelError as me:  # Catch specific model errors raised intentionally
        logger.error(f"Model error creating loss function: {me}", exc_info=False)
        raise  # Re-raise the original ModelError

    except ValueError as ve:  # Catch potential ValueErrors during loss init
        logger.error(f"Value error during loss creation: {ve}", exc_info=False)
        # Wrap as ConfigurationError as it likely stems from bad config/model state interaction
        raise ConfigurationError(
            f"Loss creation failed due to value error: {ve}"
        ) from ve

    except Exception as e:  # Catch truly unexpected errors
        logger.error(f"Unexpected error creating loss function: {e}", exc_info=True)
        # Wrap unexpected errors in ModelError as it happened during model setup phase
        raise ModelError(
            f"Unexpected failure during loss function creation: {e}"
        ) from e
