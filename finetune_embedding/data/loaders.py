# finetune_embedding/data/loaders.py
import logging
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

# Use absolute imports
from finetune_embedding.config.settings import DatasetConfig
from finetune_embedding.exceptions import DataLoadingError

logger = logging.getLogger(__name__)


class DatasetContainer:
    """Simple container to hold train, eval_dataset, and test datasets."""

    def __init__(
        self,
        train: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
    ):
        self.train_dataset = train
        self.eval_dataset = eval_dataset
        self.test_dataset = test


def _determine_load_params(
    config: DatasetConfig,
) -> Tuple[str, Dict[str, Any]]:
    """Determines the path and keyword arguments for load_dataset based on config."""
    load_path: Optional[str] = None
    load_kwargs: Dict[str, Any] = {}

    is_hub_source = (
        config.dataset_name
        and not config.file_format
        and not config.data_files
        and not config.data_dir
    )
    is_local_source = config.file_format or (
        config.dataset_name and (config.data_files or config.data_dir)
    )

    if is_hub_source:
        logger.info(f"Loading dataset '{config.dataset_name}' from Hugging Face Hub.")
        load_path = config.dataset_name
        if config.dataset_config_name:
            load_kwargs["name"] = config.dataset_config_name
    elif is_local_source:
        format_type = config.file_format or config.dataset_name
        if not format_type:
            raise DataLoadingError(
                "Cannot determine local file format. Set --file_format or use --dataset_name as format type."
            )
        logger.info(f"Loading local dataset with format '{format_type}'.")
        load_path = format_type
        if config.data_files:
            logger.info(f"Using data files: {config.data_files}")
            load_kwargs["data_files"] = config.data_files
        elif config.data_dir:
            logger.info(f"Using data directory: {config.data_dir}")
            load_kwargs["data_dir"] = config.data_dir
        else:
            raise DataLoadingError(
                f"Loading format '{format_type}' requires --data_files or --data_dir."
            )
    else:
        raise DataLoadingError("Invalid dataset source configuration provided.")

    if not load_path:
        # This case should ideally be caught by the logic above, but added for safety
        raise DataLoadingError("Could not determine dataset path or type to load.")

    return load_path, load_kwargs


def _process_loaded_dataset(dataset: Any) -> DatasetDict:
    """Validates and processes the object returned by load_dataset."""
    if not dataset:
        raise ValueError("load_dataset returned an empty object.")

    if isinstance(dataset, Dataset):
        logger.info(
            f"Loaded single Dataset ({len(dataset)} examples). Treating as 'train' initially."
        )
        dataset = DatasetDict({"train": dataset})
    elif not isinstance(dataset, DatasetDict):
        raise ValueError(f"load_dataset returned unexpected type: {type(dataset)}")

    logger.info(f"Raw dataset loaded. Initial splits: {list(dataset.keys())}")
    if dataset:  # Check if dataset is not empty
        try:
            first_split_key = list(dataset.keys())[0]
            if dataset[first_split_key]:  # Check if the first split is not empty
                logger.debug(f"Initial features: {dataset[first_split_key].features}")
            else:
                logger.warning(f"First split '{first_split_key}' is empty.")
        except IndexError:
            logger.warning("DatasetDict has no splits.")
    return dataset


def load_raw_datasets(config: DatasetConfig, cache_dir: Optional[str]) -> DatasetDict:
    """Loads dataset from Hub or local based on config."""
    logger.info("Loading raw datasets...")
    load_args = {"cache_dir": cache_dir}
    load_path: Optional[str] = None
    load_kwargs: Dict[str, Any] = {}

    try:
        # 1. Determine loading parameters
        load_path, load_kwargs = _determine_load_params(config)

        # 2. Call load_dataset
        logger.info(
            f"Calling load_dataset(path='{load_path}', split=None, **{load_args}, **{load_kwargs})"
        )
        raw_dataset = load_dataset(
            path=load_path, split=None, **load_args, **load_kwargs
        )

        # 3. Process and validate the result
        processed_dataset = _process_loaded_dataset(raw_dataset)
        return processed_dataset

    except FileNotFoundError as e:
        logger.error(
            f"Dataset loading failed: File/Dataset not found. Check paths/names. Details: {e}",
            exc_info=True,
        )
        # Try to provide the most relevant path info in the error
        source_info = load_path or config.data_files or config.data_dir or "unknown"
        raise DataLoadingError(f"Dataset file or path not found: {source_info}") from e
    except (DataLoadingError, ValueError) as e:
        # Re-raise known errors from helpers or processing
        logger.error(f"Dataset loading configuration or processing error: {e}")
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(
            f"Dataset loading failed unexpectedly. Check arguments/source. Details: {e}",
            exc_info=True,
        )
        raise DataLoadingError(f"Dataset loading failed unexpectedly: {e}") from e
