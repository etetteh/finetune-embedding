# finetune_embedding/data/hnm.py
import logging
from typing import Any, Dict, Optional, Tuple

# Import Dataset type hint if not already present
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

# Use absolute imports
from finetune_embedding.config.settings import DatasetConfig, HNMConfig
from finetune_embedding.exceptions import (
    ConfigurationError,
    DataLoadingError,
    FineTuningError,
)

# Use relative imports for sibling modules
from .loaders import DatasetContainer
from .preprocessing import get_expected_columns

logger = logging.getLogger(__name__)


def _mine_single_split(
    dataset: Optional[Dataset],
    hnm_args: Dict[str, Any],
    split_name: str,
) -> Optional[Dataset]:
    """Mines hard negatives for a single dataset split."""
    if not dataset or len(dataset) == 0:
        logger.info(
            f"No {split_name} set provided or empty, skipping HNM for {split_name}."
        )
        return None

    logger.info(f"Mining hard negatives for {split_name} set...")
    original_len = len(dataset)
    try:
        mined_dataset = mine_hard_negatives(dataset=dataset, **hnm_args)
        mined_len = len(mined_dataset)
        logger.info(
            f"{split_name.capitalize()} set after HNM: {mined_len} triplets (from {original_len} pairs)."
        )

        if mined_len == 0:
            logger.warning(
                f"Hard Negative Mining resulted in an empty {split_name} set. Setting to None."
            )
            return None
        return mined_dataset

    except Exception as e:
        # Catching exception here allows processing other splits if one fails,
        # but it might be better to let it propagate up if any failure is critical.
        # For now, log and return None for this split.
        logger.error(
            f"Failed to mine hard negatives for {split_name} set: {e}", exc_info=True
        )
        return None


def apply_hard_negative_mining(
    datasets: DatasetContainer,
    model: SentenceTransformer,
    hnm_config: HNMConfig,
    train_batch_size: int,
    dataset_config: DatasetConfig,
) -> Tuple[DatasetContainer, str]:
    """
    Applies Hard Negative Mining to train, eval, and test datasets if they exist.
    Assumes input format is 'pair'. Returns updated DatasetContainer and new effective format ('triplet').
    """
    logger.info("Applying Hard Negative Mining...")

    # --- Initial Checks ---
    if dataset_config.dataset_format != "pair":
        logger.warning(
            f"Attempted HNM but initial format is '{dataset_config.dataset_format}', not 'pair'. Skipping."
        )
        return datasets, dataset_config.dataset_format

    if not model:
        raise FineTuningError("Model must be initialized before Hard Negative Mining.")
    if not datasets.train_dataset:
        raise DataLoadingError(
            "Training data must be loaded before Hard Negative Mining."
        )

    # --- Prepare HNM Arguments ---
    try:
        initial_pair_columns = get_expected_columns("pair")
        anchor_col = initial_pair_columns["sentence1"]
        positive_col = initial_pair_columns["sentence2"]
    except Exception as e:
        raise ConfigurationError(
            f"Could not determine input columns for HNM: {e}"
        ) from e

    hnm_args = {
        "model": model,
        "anchor_column_name": anchor_col,
        "positive_column_name": positive_col,
        "num_negatives": hnm_config.num_negatives,
        "margin": hnm_config.margin,
        "range_min": hnm_config.range_min,
        "range_max": hnm_config.range_max,
        "max_score": hnm_config.max_score,
        "min_score": hnm_config.min_score,
        "sampling_strategy": hnm_config.sampling_strategy,
        "use_faiss": hnm_config.use_faiss,
        "batch_size": train_batch_size,
        "output_format": "triplet",
        "verbose": logger.isEnabledFor(logging.INFO),
    }
    logger.debug(
        f"HNM arguments: { {k: v for k, v in hnm_args.items() if k != 'model'} }"
    )

    # --- Mine Each Split ---
    try:
        mined_train_dataset = _mine_single_split(
            datasets.train_dataset, hnm_args, "training"
        )
        # Training set is critical, raise error if mining failed or resulted in empty set
        if mined_train_dataset is None:
            raise DataLoadingError(
                "Hard Negative Mining failed or resulted in an empty training set."
            )

        mined_eval_dataset = _mine_single_split(
            datasets.eval_dataset, hnm_args, "evaluation"
        )
        mined_test_dataset = _mine_single_split(datasets.test_dataset, hnm_args, "test")

        new_datasets = DatasetContainer(
            train=mined_train_dataset,
            eval_dataset=mined_eval_dataset,
            test=mined_test_dataset,
        )

        effective_format = "triplet"
        logger.info(
            f"Hard Negative Mining completed. Effective format is now '{effective_format}'."
        )
        return new_datasets, effective_format

    except DataLoadingError:
        # Re-raise critical error for training set
        raise
    except Exception as e:
        # Catch any other unexpected errors during the overall process
        logger.error(f"Hard negative mining process failed: {e}", exc_info=True)
        raise FineTuningError("Hard Negative Mining process failed.") from e
