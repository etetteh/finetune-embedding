# finetune_embedding/data/preprocessing.py
import json
import logging
import os
from typing import Any, Dict, Optional, Union

from datasets import Dataset, DatasetDict

# Use absolute imports
from finetune_embedding.config.settings import (
    DEFAULT_EVAL_PROP,
    DEFAULT_TEST_PROP,
    DEFAULT_TRAIN_PROP,
    DatasetConfig,
)
from finetune_embedding.exceptions import ConfigurationError, DataLoadingError

# Use relative import for sibling module
from .loaders import DatasetContainer

logger = logging.getLogger(__name__)

# Define expected column names for clarity and easier maintenance
COLUMNS_TRIPLET = {"anchor": "anchor", "positive": "positive", "negative": "negative"}
COLUMNS_PAIR_SCORE = {
    "sentence1": "sentence1",
    "sentence2": "sentence2",
    "score": "score",
}
COLUMNS_PAIR = {"sentence1": "sentence1", "sentence2": "sentence2"}  # Generic pair
COLUMNS_PAIR_CLASS = {
    "sentence1": "sentence1",
    "sentence2": "sentence2",
    "label": "label",
}


def get_expected_columns(dataset_format: str) -> Dict[str, str]:
    """Gets the standard column names for a given format."""
    if dataset_format == "triplet":
        return COLUMNS_TRIPLET.copy()
    if dataset_format == "pair-score":
        return COLUMNS_PAIR_SCORE.copy()
    if dataset_format == "pair":
        return COLUMNS_PAIR.copy()
    if dataset_format == "pair-class":
        return COLUMNS_PAIR_CLASS.copy()
    raise ConfigurationError(
        f"Unsupported dataset_format '{dataset_format}' for column selection."
    )


def parse_column_rename_map(
    map_input: Optional[Union[str, Dict, Any]],
) -> Optional[Dict[str, str]]:
    """Parses the column rename map from config (JSON string, file path, or dict)."""
    if not map_input:
        return None

    rename_map: Dict[str, Any]

    if isinstance(map_input, dict):
        rename_map = map_input
        logger.debug("Using pre-parsed dictionary for column_rename_map.")
    elif isinstance(map_input, str):
        try:  # Try as JSON string
            rename_map = json.loads(map_input)
            logger.debug("Parsed column_rename_map as JSON string.")
        except json.JSONDecodeError:  # Try as file path
            if os.path.isfile(map_input):
                try:
                    with open(map_input, "r") as f:
                        rename_map = json.load(f)
                    logger.debug(f"Loaded column_rename_map from file: {map_input}")
                except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    raise ConfigurationError(
                        f"Failed to load/parse column rename map file '{map_input}': {e}"
                    ) from e
            else:
                raise ConfigurationError(
                    f"'{map_input}' is neither valid JSON, a dict, nor an existing file."
                ) from None
    else:
        raise ConfigurationError(
            f"Unsupported type for column_rename_map: {type(map_input)}. Expected str or dict."
        )

    # Validation
    if not isinstance(rename_map, dict):
        raise ConfigurationError(
            f"Column rename map must resolve to a dictionary, got {type(rename_map)}."
        )

    validated_map: Dict[str, str] = {}
    for key, value in rename_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ConfigurationError(
                f"Keys/values in column_rename_map must be strings. Found: {key}({type(key)})={value}({type(value)})"
            )
        validated_map[key] = value

    logger.info(f"Using column rename map: {validated_map}")
    return validated_map


def apply_column_renaming(
    dataset_dict: DatasetDict, rename_map: Optional[Dict[str, str]]
) -> DatasetDict:
    """Applies column renaming to all splits in a DatasetDict."""
    if not rename_map:
        logger.info("No column rename map provided, skipping renaming.")
        return dataset_dict

    logger.info("Applying column renaming...")
    try:
        renamed_dataset_dict = DatasetDict()
        for split_name, split_data in dataset_dict.items():
            current_columns = set(split_data.column_names)
            map_for_split = {
                k: v for k, v in rename_map.items() if k in current_columns
            }
            missing_source_keys = set(rename_map.keys()) - current_columns
            if missing_source_keys:
                logger.warning(
                    f"Rename source keys {missing_source_keys} not found in split '{split_name}'."
                )

            if map_for_split:
                renamed_dataset_dict[split_name] = split_data.rename_columns(
                    map_for_split
                )
                logger.debug(f"Renamed columns in '{split_name}': {map_for_split}")
            else:
                renamed_dataset_dict[split_name] = (
                    split_data  # No changes needed for this split
                )
        logger.info("Column renaming applied.")
        if renamed_dataset_dict:  # Check if not empty
            first_split_key = list(renamed_dataset_dict.keys())[0]
            if renamed_dataset_dict[first_split_key]:  # Check if split not empty
                logger.debug(
                    f"Features after rename: {renamed_dataset_dict[first_split_key].features}"
                )
        return renamed_dataset_dict

    except Exception as e:
        logger.error(f"Failed during column renaming: {e}", exc_info=True)
        raise DataLoadingError(f"Column renaming failed: {e}") from e


def auto_split_dataset(
    dataset_dict: DatasetDict, config: DatasetConfig, seed: int
) -> DatasetDict:
    """Automatically splits a single-split dataset into train/eval/test."""
    if len(dataset_dict) > 1:
        logger.info("Multiple splits found, skipping auto-split.")
        return dataset_dict

    single_split_name = list(dataset_dict.keys())[0]
    source_data = dataset_dict[single_split_name]
    source_len = len(source_data)
    logger.info(
        f"Only one split ('{single_split_name}', {source_len} examples) loaded. Attempting auto-split."
    )

    min_required_for_split = 10
    if source_len < min_required_for_split:
        logger.warning(
            f"Dataset too small ({source_len} < {min_required_for_split}) for 3-way split. Using all for training split '{config.train_split}'."
        )
        return DatasetDict({config.train_split: source_data})

    try:
        logger.info(
            f"Auto-splitting with proportions: Train={DEFAULT_TRAIN_PROP}, Eval={DEFAULT_EVAL_PROP}, Test={DEFAULT_TEST_PROP} (Seed: {seed})"
        )
        test_prop = DEFAULT_TEST_PROP
        train_eval_prop = 1.0 - test_prop
        train_eval_split = source_data.train_test_split(
            test_size=test_prop, shuffle=True, seed=seed
        )
        temp_train_eval_data = train_eval_split["train"]
        test_data_auto = train_eval_split["test"]

        eval_prop_relative = DEFAULT_EVAL_PROP / train_eval_prop
        if not (0.0 < eval_prop_relative < 1.0):
            logger.warning(
                f"Calculated relative eval proportion ({eval_prop_relative:.2f}) is invalid. Setting to 0.1."
            )
            eval_prop_relative = 0.1

        final_split = temp_train_eval_data.train_test_split(
            test_size=eval_prop_relative, shuffle=True, seed=seed
        )
        train_data_auto = final_split["train"]
        eval_data_auto = final_split["test"]

        final_dataset_dict = DatasetDict(
            {
                config.train_split: train_data_auto,
                config.eval_split: eval_data_auto,
                config.test_split: test_data_auto,
            }
        )
        logger.info(
            f"Auto-split successful. Final splits: { {k: len(v) for k, v in final_dataset_dict.items()} }"
        )
        return final_dataset_dict

    except ValueError as e:
        logger.error(
            f"Automatic splitting failed (likely due to small dataset size): {e}. Using original single split for training split '{config.train_split}' only.",
            exc_info=False,
        )
        return DatasetDict({config.train_split: source_data})


def select_and_limit_splits(
    dataset_dict: DatasetDict, config: DatasetConfig
) -> DatasetContainer:
    """Selects train, eval, test splits based on config, applies limits, and returns a container."""
    logger.info("Selecting and limiting dataset splits...")
    available_split_names = list(dataset_dict.keys())
    train_ds, eval_ds, test_ds = None, None, None

    user_specified_eval_split = config.eval_split
    target_eval_split_name = user_specified_eval_split
    common_eval_names = ["dev", "validation", "eval", "val"]

    if target_eval_split_name not in available_split_names:
        logger.warning(
            f"Configured eval split '{target_eval_split_name}' not found in available splits: {available_split_names}. Attempting auto-detection..."
        )
        found_alternative = False
        for potential_name in common_eval_names:
            if potential_name in available_split_names:
                target_eval_split_name = potential_name
                logger.info(
                    f"Automatically selected existing split '{target_eval_split_name}' for evaluation."
                )
                found_alternative = True
                break
        if not found_alternative:
            logger.warning(
                f"Could not find any common evaluation splits ({common_eval_names}) either. Evaluation data may be unavailable."
            )
    else:
        logger.info(f"Using specified eval split name: '{target_eval_split_name}'")

    def _select_limit(
        split_key: str, limit: int, available_dict: DatasetDict
    ) -> Optional[Dataset]:
        if split_key not in available_dict:
            logger.warning(
                f"Split '{split_key}' not found in available splits: {list(available_dict.keys())}. Skipping this split."
            )
            return None
        data = available_dict[split_key]
        orig_len = len(data)
        if orig_len == 0:
            logger.warning(f"Split '{split_key}' is empty. Skipping.")
            return None
        if limit > 0 and limit < orig_len:
            logger.info(
                f"Limiting '{split_key}' split from {orig_len} to {limit} examples."
            )
            return data.select(range(limit))
        elif limit > 0:
            logger.info(
                f"Limit ({limit}) >= available ({orig_len}) for '{split_key}'. Using all examples."
            )
            return data
        else:
            logger.info(f"Using full '{split_key}' split ({orig_len} examples).")
            return data

    train_ds = _select_limit(config.train_split, config.train_limit, dataset_dict)
    eval_ds = _select_limit(target_eval_split_name, config.eval_limit, dataset_dict)
    test_ds = _select_limit(config.test_split, config.test_limit, dataset_dict)

    if train_ds is None or len(train_ds) == 0:
        raise DataLoadingError(
            "Training dataset is empty after processing. Check config/limits/data."
        )
    if eval_ds is None or len(eval_ds) == 0:
        logger.warning(
            "Evaluation dataset is empty or unavailable. Evaluation during training will be skipped."
        )
    if test_ds is None or len(test_ds) == 0:
        logger.warning(
            "Test dataset is empty or unavailable. Final evaluation on test set will be skipped."
        )

    logger.info(
        f"Final dataset sizes: "
        f"Train={len(train_ds) if train_ds else 0}, "
        f"Eval={len(eval_ds) if eval_ds else 0} (used split name: '{target_eval_split_name if eval_ds else 'N/A'}'), "
        f"Test={len(test_ds) if test_ds else 0}"
    )

    # Validate Required Columns after all processing
    try:
        expected_cols = get_expected_columns(config.dataset_format)
        final_train_cols = set(train_ds.column_names)
        missing_req_cols = set(expected_cols.values()) - final_train_cols
        if missing_req_cols:
            raise DataLoadingError(
                f"Train dataset missing required columns for format '{config.dataset_format}'. Required: {expected_cols.values()}, Missing: {missing_req_cols}. Available: {final_train_cols}"
            )
        logger.info("Required columns verified in training data.")
    except (
        ConfigurationError
    ) as e:  # Catch error from get_expected_columns if format is bad
        raise DataLoadingError(
            f"Cannot verify columns due to invalid format: {e}"
        ) from e

    return DatasetContainer(train=train_ds, eval_dataset=eval_ds, test=test_ds)
