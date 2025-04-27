# finetune_embedding/data/loaders.py
import logging
from typing import Any, Dict, Optional

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


def load_raw_datasets(config: DatasetConfig, cache_dir: Optional[str]) -> DatasetDict:
    """Loads dataset from Hub or local based on config."""
    logger.info("Loading raw datasets...")
    load_args = {"cache_dir": cache_dir}
    load_path: Optional[str] = None
    load_kwargs: Dict[str, Any] = {}

    # Determine loading method (Hub vs Local)
    if (
        config.dataset_name
        and not config.file_format
        and not config.data_files
        and not config.data_dir
    ):
        logger.info(f"Loading dataset '{config.dataset_name}' from Hugging Face Hub.")
        load_path = config.dataset_name
        if config.dataset_config_name:
            load_kwargs["name"] = config.dataset_config_name
    elif config.file_format or (
        config.dataset_name and (config.data_files or config.data_dir)
    ):
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

    logger.info(
        f"Calling load_dataset(path='{load_path}', split=None, **{load_args}, **{load_kwargs})"
    )
    try:
        dataset = load_dataset(path=load_path, split=None, **load_args, **load_kwargs)

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
            first_split_key = list(dataset.keys())[0]
            if dataset[first_split_key]:  # Check if the first split is not empty
                logger.debug(f"Initial features: {dataset[first_split_key].features}")
            else:
                logger.warning(f"First split '{first_split_key}' is empty.")
        return dataset

    except FileNotFoundError as e:
        logger.error(
            f"Dataset loading failed: File/Dataset not found. Check paths/names. Details: {e}",
            exc_info=True,
        )
        raise DataLoadingError(
            f"Dataset file or path not found: {load_path or config.data_files or config.data_dir}"
        ) from e
    except Exception as e:
        logger.error(
            f"Dataset loading failed unexpectedly. Check arguments/source. Details: {e}",
            exc_info=True,
        )
        raise DataLoadingError(f"Dataset loading failed: {e}") from e
