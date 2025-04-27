# finetune_embedding/training/losses.py
import logging

from sentence_transformers import SentenceTransformer, losses

# Use absolute imports
from finetune_embedding.exceptions import ConfigurationError, ModelError

logger = logging.getLogger(__name__)


def create_loss_function(
    model: SentenceTransformer,
    effective_format: str,
    num_labels: int,  # Required for SoftmaxLoss
) -> object:  # Return type depends on the specific loss class
    """Creates the appropriate loss function based on the effective dataset format."""
    logger.info(f"Creating loss function for effective format '{effective_format}'.")
    if not model:
        raise ModelError("Model must be initialized before creating loss.")

    try:
        if effective_format == "triplet":
            loss = losses.MultipleNegativesRankingLoss(model)
            logger.info("Selected MultipleNegativesRankingLoss.")
        elif effective_format == "pair-score":
            loss = losses.CoSENTLoss(model)
            logger.info("Selected CoSENTLoss.")
        elif effective_format == "pair-class":
            if num_labels <= 1:
                # Raise the specific configuration error
                raise ConfigurationError("SoftmaxLoss requires num_labels > 1.")
            emb_dim = model.get_sentence_embedding_dimension()
            if not emb_dim:
                # Raise ModelError if dimension cannot be determined
                raise ModelError(
                    "Could not determine embedding dimension for SoftmaxLoss."
                )
            loss = losses.SoftmaxLoss(
                model=model,
                sentence_embedding_dimension=emb_dim,
                num_labels=num_labels,
            )
            logger.info(
                f"Selected SoftmaxLoss with num_labels={num_labels}, embedding_dim={emb_dim}."
            )
        else:
            # Raise the specific configuration error for unsupported format
            raise ConfigurationError(
                f"Unsupported effective format '{effective_format}' for loss creation."
            )

        logger.info(f"Loss function created: {type(loss).__name__}")
        return loss

    # --- Corrected Exception Handling Order ---
    except (
        ConfigurationError
    ) as ce:  # Catch specific config errors raised intentionally above
        logger.error(
            f"Configuration error creating loss function: {ce}", exc_info=False
        )
        raise  # Re-raise the original ConfigurationError

    except ModelError as me:  # Catch specific model errors raised intentionally above
        logger.error(f"Model error creating loss function: {me}", exc_info=False)
        raise  # Re-raise the original ModelError

    # Catch other potential ValueErrors (e.g., from SentenceTransformer methods if they raise it)
    except ValueError as ve:
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


# # finetune_embedding/training/losses.py
# import logging
# from sentence_transformers import SentenceTransformer, losses

# # Use absolute imports
# from finetune_embedding.exceptions import ConfigurationError, ModelError

# logger = logging.getLogger(__name__)

# def create_loss_function(
#     model: SentenceTransformer,
#     effective_format: str,
#     num_labels: int # Required for SoftmaxLoss
# ) -> object: # Return type depends on the specific loss class
#     """Creates the appropriate loss function based on the effective dataset format."""
#     logger.info(f"Creating loss function for effective format '{effective_format}'.")
#     if not model:
#         raise ModelError("Model must be initialized before creating loss.")

#     try:
#         if effective_format == "triplet":
#             loss = losses.MultipleNegativesRankingLoss(model)
#             logger.info(f"Selected MultipleNegativesRankingLoss.")
#         elif effective_format == "pair-score":
#             loss = losses.CoSENTLoss(model)
#             logger.info("Selected CoSENTLoss.")
#         elif effective_format == "pair-class":
#             if num_labels <= 1:
#                 raise ConfigurationError("SoftmaxLoss requires num_labels > 1.")
#             emb_dim = model.get_sentence_embedding_dimension()
#             if not emb_dim:
#                 raise ModelError("Could not determine embedding dimension for SoftmaxLoss.")
#             loss = losses.SoftmaxLoss(
#                 model=model,
#                 sentence_embedding_dimension=emb_dim,
#                 num_labels=num_labels,
#             )
#             logger.info(f"Selected SoftmaxLoss with num_labels={num_labels}, embedding_dim={emb_dim}.")
#         else:
#             raise ConfigurationError(f"Unsupported effective format '{effective_format}' for loss creation.")

#         logger.info(f"Loss function created: {type(loss).__name__}")
#         return loss

#     except ValueError as ve:
#         logger.error(f"Failed to create loss function: {ve}", exc_info=False)
#         raise ConfigurationError(f"Loss creation failed: {ve}") from ve
#     except Exception as e:
#         logger.error(f"Unexpected error creating loss function: {e}", exc_info=True)
#         raise ModelError(f"Failed to create loss function: {e}") from e # Changed to ModelError as it relates to model state
