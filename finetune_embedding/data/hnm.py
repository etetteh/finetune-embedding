# finetune_embedding/data/hnm.py
import logging
from typing import Tuple

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


def apply_hard_negative_mining(
    datasets: DatasetContainer,
    model: SentenceTransformer,
    hnm_config: HNMConfig,
    train_batch_size: int,
    dataset_config: DatasetConfig,  # Pass dataset config for column names
) -> Tuple[DatasetContainer, str]:
    """
    Applies Hard Negative Mining to train, eval, and test datasets if they exist.
    Assumes input format is 'pair'. Returns updated DatasetContainer and new effective format ('triplet').
    """
    logger.info("Applying Hard Negative Mining...")

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

    new_datasets = DatasetContainer()

    try:
        logger.info("Mining hard negatives for training set...")
        original_len = len(datasets.train_dataset)
        new_datasets.train_dataset = mine_hard_negatives(
            dataset=datasets.train_dataset, **hnm_args
        )
        mined_len = len(new_datasets.train_dataset)
        logger.info(
            f"Training set after HNM: {mined_len} triplets (from {original_len} pairs)."
        )
        if mined_len == 0:
            raise DataLoadingError(
                "Hard Negative Mining resulted in an empty training set."
            )

        if datasets.eval_dataset and len(datasets.eval_dataset) > 0:
            logger.info("Mining hard negatives for evaluation set...")
            original_len = len(datasets.eval_dataset)
            new_datasets.eval_dataset = mine_hard_negatives(
                dataset=datasets.eval_dataset, **hnm_args
            )
            mined_len = len(new_datasets.eval_dataset)
            logger.info(
                f"Evaluation set after HNM: {mined_len} triplets (from {original_len} pairs)."
            )
            if mined_len == 0:
                new_datasets.eval_dataset = None
        else:
            logger.info("No evaluation set provided or empty, skipping HNM for eval.")
            new_datasets.eval_dataset = None

        if datasets.test_dataset and len(datasets.test_dataset) > 0:
            logger.info("Mining hard negatives for test set...")
            original_len = len(datasets.test_dataset)
            new_datasets.test_dataset = mine_hard_negatives(
                dataset=datasets.test_dataset, **hnm_args
            )
            mined_len = len(new_datasets.test_dataset)
            logger.info(
                f"Test set after HNM: {mined_len} triplets (from {original_len} pairs)."
            )
            if mined_len == 0:
                new_datasets.test_dataset = None
        else:
            logger.info("No test set provided or empty, skipping HNM for test.")
            new_datasets.test_dataset = None

        effective_format = "triplet"
        logger.info(
            f"Hard Negative Mining completed. Effective format is now '{effective_format}'."
        )
        return new_datasets, effective_format

    except Exception as e:
        logger.error(f"Hard negative mining failed: {e}", exc_info=True)
        raise FineTuningError("Hard Negative Mining failed.") from e
