# finetune_embedding/evaluation/evaluators.py
import logging
from typing import Callable, Dict, Optional

from datasets import Dataset
from sentence_transformers import evaluation

from finetune_embedding.config.settings import DatasetConfig
from finetune_embedding.data.preprocessing import get_expected_columns
from finetune_embedding.exceptions import (
    ConfigurationError,
    EvaluationError,
)

logger = logging.getLogger(__name__)

# Type alias for evaluator creation functions
EvaluatorCreator = Callable[
    [Dataset, Dict[str, str], str, int],
    Optional[evaluation.SentenceEvaluator],
]


def _create_triplet_evaluator(
    eval_dataset: Dataset,
    cols: Dict[str, str],
    eval_name: str,
    eval_batch_size: int,
) -> Optional[evaluation.SentenceEvaluator]:
    """Creates a TripletEvaluator."""
    try:
        return evaluation.TripletEvaluator(
            anchors=eval_dataset[cols["anchor"]],
            positives=eval_dataset[cols["positive"]],
            negatives=eval_dataset[cols["negative"]],
            name=eval_name,
            batch_size=eval_batch_size,
            show_progress_bar=True,
        )
    except KeyError as e:
        raise ConfigurationError(
            f"Missing required column for TripletEvaluator: {e}. Required: {cols.values()}, Available: {eval_dataset.column_names}"
        ) from e


def _create_similarity_evaluator(
    eval_dataset: Dataset,
    cols: Dict[str, str],
    eval_name: str,
    eval_batch_size: int,
) -> Optional[evaluation.SentenceEvaluator]:
    """Creates an EmbeddingSimilarityEvaluator."""
    try:
        return evaluation.EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset[cols["sentence1"]],
            sentences2=eval_dataset[cols["sentence2"]],
            scores=eval_dataset[cols["score"]],
            name=eval_name,
            batch_size=eval_batch_size,
            show_progress_bar=True,
        )
    except KeyError as e:
        raise ConfigurationError(
            f"Missing required column for EmbeddingSimilarityEvaluator: {e}. Required: {cols.values()}, Available: {eval_dataset.column_names}"
        ) from e


def _create_binary_classification_evaluator(
    eval_dataset: Dataset,
    cols: Dict[str, str],
    eval_name: str,
    eval_batch_size: int,
    name_prefix: str = "eval",  # Pass prefix for logging
) -> Optional[evaluation.SentenceEvaluator]:
    """Creates a BinaryClassificationEvaluator, filtering for binary labels."""
    try:
        s1_col, s2_col, label_col = (
            cols["sentence1"],
            cols["sentence2"],
            cols["label"],
        )
    except KeyError as e:
        raise ConfigurationError(
            f"Missing required column for BinaryClassificationEvaluator: {e}. Required: {cols.values()}, Available: {eval_dataset.column_names}"
        ) from e

    original_len = len(eval_dataset)
    # Ensure labels are integers for filtering
    try:
        labels = [int(lbl) for lbl in eval_dataset[label_col]]
    except (ValueError, TypeError) as label_err:
        raise ConfigurationError(
            f"Labels in column '{label_col}' for BinaryClassificationEvaluator must be convertible to integers."
        ) from label_err

    # Filter for binary labels (0 or 1)
    valid_indices = [i for i, lbl in enumerate(labels) if lbl in [0, 1]]

    if not valid_indices:
        logger.error(
            f"Cannot create BinaryClassificationEvaluator '{eval_name}': No examples with labels 0 or 1 found in '{label_col}'."
        )
        return None  # Return None if no valid data

    filtered_dataset = eval_dataset.select(valid_indices)
    filtered_len = len(filtered_dataset)

    if filtered_len < original_len:
        logger.warning(
            f"Discarded {original_len - filtered_len} examples from {name_prefix} set '{eval_name}' (labels not 0/1 in '{label_col}')."
        )

    logger.info(
        f"Using {filtered_len} examples for BinaryClassificationEvaluator '{eval_name}'."
    )

    return evaluation.BinaryClassificationEvaluator(
        sentences1=filtered_dataset[s1_col],
        sentences2=filtered_dataset[s2_col],
        labels=[labels[i] for i in valid_indices],  # Use filtered integer labels
        name=eval_name,
        batch_size=eval_batch_size,
        show_progress_bar=True,
        write_csv=True,
    )


# --- Dispatch Dictionary ---
EVALUATOR_CREATORS: Dict[str, EvaluatorCreator] = {
    "triplet": _create_triplet_evaluator,
    "pair-score": _create_similarity_evaluator,
    "pair-class": _create_binary_classification_evaluator,
    # Add other formats here if needed
}


# --- Refactored Main Function ---
def create_evaluator(
    eval_dataset: Optional[Dataset],
    effective_format: str,
    dataset_config: DatasetConfig,
    eval_batch_size: int,
    name_prefix: str = "eval",  # e.g., "eval" or "test"
) -> Optional[evaluation.SentenceEvaluator]:
    """Creates the appropriate evaluator based on the effective dataset format using dispatch."""
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.info(
            f"No {name_prefix} data available. Skipping {name_prefix} evaluator creation."
        )
        return None

    logger.info(
        f"Creating {name_prefix} evaluator for effective format '{effective_format}'."
    )

    # --- Prepare Common Arguments ---
    eval_name = "unknown_evaluator"  # Default name in case of early error
    try:
        cols = get_expected_columns(effective_format)

        dataset_name_part = (
            dataset_config.dataset_name.split("/")[-1]
            if dataset_config.dataset_name
            else "local"
        )
        split_name_part = (
            dataset_config.eval_split
            if name_prefix == "eval"
            else dataset_config.test_split
        )
        eval_name = f"{name_prefix}_{dataset_name_part}-{split_name_part}"

        # --- Check for Missing Columns ---
        missing_cols = set(cols.values()) - set(eval_dataset.column_names)
        if missing_cols:
            raise ConfigurationError(
                f"{name_prefix.capitalize()} dataset missing required columns for format '{effective_format}': {missing_cols}. Available: {eval_dataset.column_names}"
            )

        # --- Dispatch to Specific Creator ---
        creator_func = EVALUATOR_CREATORS.get(effective_format)
        if not creator_func:
            # This case should ideally be caught by config validation earlier
            raise ConfigurationError(
                f"Unsupported effective_format '{effective_format}' for evaluator creation."
            )

        # Pass name_prefix to binary creator if needed for logging
        if effective_format == "pair-class":
            evaluator = creator_func(
                eval_dataset, cols, eval_name, eval_batch_size, name_prefix=name_prefix
            )
        else:
            evaluator = creator_func(eval_dataset, cols, eval_name, eval_batch_size)

        # --- Log Result ---
        if evaluator:
            logger.info(
                f"Successfully created {type(evaluator).__name__} evaluator for '{eval_name}'."
            )
        else:
            # This case is now only hit if a creator function explicitly returns None (e.g., binary class with no valid labels)
            logger.warning(
                f"Could not create {name_prefix} evaluator for '{eval_name}' (e.g., no valid binary labels found)."
            )
        return evaluator

    # --- Error Handling ---
    except (KeyError, ValueError, ConfigurationError) as specific_error:
        # Catch expected errors related to columns, values, or config
        logger.error(
            f"Failed creating {name_prefix} evaluator '{eval_name}': {specific_error}",
            exc_info=False,  # Keep False for cleaner logs unless debugging
        )
        # Re-raise the specific error caught
        raise specific_error
    except Exception as e:
        # Catch truly unexpected errors
        logger.error(
            f"Unexpected error creating {name_prefix} evaluator '{eval_name}': {e}",
            exc_info=True,
        )
        raise EvaluationError(
            f"Unexpected error creating {name_prefix} evaluator '{eval_name}': {e}"
        ) from e
