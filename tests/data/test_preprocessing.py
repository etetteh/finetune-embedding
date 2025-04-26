# tests/data/test_preprocessing.py
import pytest
import json
import os
import logging # Import logging
from pathlib import Path
from unittest.mock import patch, mock_open
from datasets import Dataset, DatasetDict

# Use absolute imports
from finetune_embedding.data.preprocessing import (
    get_expected_columns,
    parse_column_rename_map,
    apply_column_renaming,
    auto_split_dataset,
    select_and_limit_splits
)
from finetune_embedding.data.loaders import DatasetContainer
from finetune_embedding.config.settings import DatasetConfig
from finetune_embedding.exceptions import ConfigurationError, DataLoadingError

# --- Fixtures ---

@pytest.fixture
def sample_json_file_factory(tmp_path_factory):
    """Factory to create sample JSON files (re-used from conftest if available)."""
    def _factory(filename="sample_map.json", content={"orig_a": "anchor"}):
        file_path = tmp_path_factory.mktemp("config") / filename
        with open(file_path, 'w') as f:
            json.dump(content, f)
        return file_path
    return _factory

@pytest.fixture
def single_split_dict() -> DatasetDict:
    return DatasetDict({"train": Dataset.from_dict({"feature": list(range(100))})})

@pytest.fixture
def full_dataset_dict() -> DatasetDict:
    # Add required columns for 'pair' format tests
    return DatasetDict({
        "train": Dataset.from_dict({"sentence1": [f"tr{i}a" for i in range(100)], "sentence2": [f"tr{i}b" for i in range(100)]}),
        "validation": Dataset.from_dict({"sentence1": [f"va{i}a" for i in range(30)], "sentence2": [f"va{i}b" for i in range(30)]}),
        "test": Dataset.from_dict({"sentence1": [f"te{i}a" for i in range(20)], "sentence2": [f"te{i}b" for i in range(20)]}),
        "dev": Dataset.from_dict({"sentence1": [f"de{i}a" for i in range(10)], "sentence2": [f"de{i}b" for i in range(10)]}),
    })

@pytest.fixture
def pair_format_config() -> DatasetConfig:
    # Basic config for 'pair' format, adjust splits/limits in tests
    return DatasetConfig(dataset_format="pair", train_split="train", eval_split="validation", test_split="test")

@pytest.fixture
def mock_config(pair_format_config):
    # Simple mock config for auto_split test
    class MockConfig:
        train_split = "train"
        eval_split = "validation"
        test_split = "test"
        seed = 42
        # Add proportions used by auto_split
        train_proportion = 0.8
        eval_proportion = 0.1
        test_proportion = 0.1
    return MockConfig()


# --- Tests for get_expected_columns ---
@pytest.mark.parametrize("format, expected_keys", [
    ("triplet", ["anchor", "positive", "negative"]),
    ("pair", ["sentence1", "sentence2"]),
    ("pair-score", ["sentence1", "sentence2", "score"]),
    ("pair-class", ["sentence1", "sentence2", "label"]),
])
def test_get_expected_columns(format, expected_keys):
    cols = get_expected_columns(format)
    assert isinstance(cols, dict)
    assert list(cols.keys()) == expected_keys
    assert all(isinstance(v, str) for v in cols.values())

# --- Tests for parse_column_rename_map ---

def test_parse_column_rename_map_none():
    assert parse_column_rename_map(None) is None

def test_parse_column_rename_map_dict():
    input_dict = {"orig_a": "anchor", "orig_p": "positive"}
    assert parse_column_rename_map(input_dict) == input_dict

def test_parse_column_rename_map_json_string():
    input_json = '{"orig_a": "anchor", "orig_p": "positive"}'
    expected = {"orig_a": "anchor", "orig_p": "positive"}
    assert parse_column_rename_map(input_json) == expected

def test_parse_column_rename_map_invalid_json_string():
    input_json = '{"orig_a": "anchor", "orig_p": positive}' # Invalid JSON value
    # Match actual error message
    with pytest.raises(ConfigurationError, match="'{.*}' is neither valid JSON, a dict, nor an existing file."):
        parse_column_rename_map(input_json)

def test_parse_column_rename_map_file(sample_json_file_factory):
    expected = {"orig_x": "sentence1", "orig_y": "sentence2"}
    file_path = sample_json_file_factory(content=expected)
    assert parse_column_rename_map(str(file_path)) == expected

def test_parse_column_rename_map_file_not_found():
     # Match actual error message
     with pytest.raises(ConfigurationError, match="'non_existent_map.json' is neither valid JSON, a dict, nor an existing file."):
         parse_column_rename_map("non_existent_map.json")

def test_parse_column_rename_map_invalid_type():
    with pytest.raises(ConfigurationError, match="Unsupported type for column_rename_map"):
        parse_column_rename_map(123)

def test_parse_column_rename_map_invalid_content():
    # Valid JSON, but not a dictionary
    with pytest.raises(ConfigurationError, match="must resolve to a dictionary"):
        parse_column_rename_map('[1, 2, 3]')
    # Valid dictionary, but invalid key/value types
    with pytest.raises(ConfigurationError, match="Keys/values in column_rename_map must be strings"):
        parse_column_rename_map({"key": 1})

# --- Tests for apply_column_renaming ---

def test_apply_column_renaming_no_map():
    ds = Dataset.from_dict({"colA": [1], "colB": [2]})
    ds_dict = DatasetDict({"train": ds})
    result = apply_column_renaming(ds_dict, None)
    assert result == ds_dict # No change
    assert list(result["train"].column_names) == ["colA", "colB"]

def test_apply_column_renaming_success():
    ds = Dataset.from_dict({"colA": [1], "colB": [2]})
    ds_dict = DatasetDict({"train": ds})
    rename_map = {"colA": "sentence1", "colB": "sentence2"}
    result = apply_column_renaming(ds_dict, rename_map)
    assert list(result["train"].column_names) == ["sentence1", "sentence2"]

def test_apply_column_renaming_missing_source_key(caplog):
    ds = Dataset.from_dict({"colA": [1]})
    ds_dict = DatasetDict({"train": ds})
    rename_map = {"colA": "sentence1", "colB": "sentence2"} # colB doesn't exist
    result = apply_column_renaming(ds_dict, rename_map)
    assert list(result["train"].column_names) == ["sentence1"] # Only colA renamed
    # --- FIX: Add "source" to expected log message ---
    assert "Rename source keys {'colB'} not found in split 'train'" in caplog.text
    # --- END FIX ---

def test_apply_column_renaming_multiple_splits():
    ds1 = Dataset.from_dict({"colA": [1], "colB": [2]})
    ds2 = Dataset.from_dict({"colA": [3], "colC": [4]}) # Different columns
    ds_dict = DatasetDict({"train": ds1, "test": ds2})
    rename_map = {"colA": "common", "colB": "b_new", "colC": "c_new"}
    result = apply_column_renaming(ds_dict, rename_map)
    assert list(result["train"].column_names) == ["common", "b_new"]
    assert list(result["test"].column_names) == ["common", "c_new"]

# --- Tests for auto_split_dataset ---

def test_auto_split_skips_multi_split(full_dataset_dict, mock_config):
    # Should return original dict if more than one split exists
    result = auto_split_dataset(full_dataset_dict, mock_config, seed=42)
    assert result == full_dataset_dict

def test_auto_split_success(single_split_dict, mock_config):
    result = auto_split_dataset(single_split_dict, mock_config, seed=42)
    assert len(result) == 3
    assert mock_config.train_split in result
    assert mock_config.eval_split in result
    assert mock_config.test_split in result
    # Assert exact lengths based on calculation
    assert len(result[mock_config.train_split]) == 80
    assert len(result[mock_config.eval_split]) == 10
    assert len(result[mock_config.test_split]) == 10

def test_auto_split_too_small(mock_config):
    small_ds = Dataset.from_dict({"feature": list(range(5))})
    small_dict = DatasetDict({"train": small_ds})
    result = auto_split_dataset(small_dict, mock_config, seed=42)
    assert len(result) == 1 # Should fallback to train only
    assert mock_config.train_split in result
    assert mock_config.eval_split not in result
    assert mock_config.test_split not in result
    assert len(result[mock_config.train_split]) == 5

# --- Tests for select_and_limit_splits ---

def test_select_limit_basic(full_dataset_dict, pair_format_config):
    config = pair_format_config
    container = select_and_limit_splits(full_dataset_dict, config)
    assert isinstance(container, DatasetContainer)
    assert container.train_dataset is not None
    assert container.eval_dataset is not None
    assert container.test_dataset is not None
    assert len(container.train_dataset) == 100 # No limit
    assert len(container.eval_dataset) == 30  # Uses 'validation' split
    assert len(container.test_dataset) == 20  # Uses 'test' split
    # Check columns are present (validation was failing here)
    assert "sentence1" in container.train_dataset.column_names
    assert "sentence2" in container.train_dataset.column_names

def test_select_limit_with_limits(full_dataset_dict, pair_format_config):
    config = pair_format_config
    config.train_limit = 10
    config.eval_limit = 5
    config.test_limit = 0 # Use all test
    container = select_and_limit_splits(full_dataset_dict, config)
    assert len(container.train_dataset) == 10
    assert len(container.eval_dataset) == 5
    assert len(container.test_dataset) == 20 # No limit applied

def test_select_limit_eval_auto_detect(full_dataset_dict, pair_format_config, caplog):
    # --- FIX: Set caplog level ---
    caplog.set_level(logging.INFO)
    # --- END FIX ---
    config = pair_format_config
    config.eval_split = "non_existent_split" # Force auto-detect
    container = select_and_limit_splits(full_dataset_dict, config)
    assert "Configured eval split 'non_existent_split' not found" in caplog.text
    assert "Automatically selected existing split 'dev'" in caplog.text # Should now be captured
    assert container.eval_dataset is not None
    assert len(container.eval_dataset) == 10 # Should have found and used 'dev'

def test_select_limit_missing_split(full_dataset_dict, pair_format_config, caplog):
    config = pair_format_config
    config.test_split = "missing_test"
    container = select_and_limit_splits(full_dataset_dict, config)
    assert "Split 'missing_test' not found" in caplog.text
    assert container.test_dataset is None # Test split should be None

def test_select_limit_empty_train_fails(pair_format_config):
    empty_dict = DatasetDict({"train": Dataset.from_dict({"sentence1": [], "sentence2": []})})
    with pytest.raises(DataLoadingError, match="Training dataset is empty"):
        select_and_limit_splits(empty_dict, pair_format_config)

def test_select_limit_missing_required_cols(pair_format_config):
    # Test the validation check directly
    bad_dict = DatasetDict({"train": Dataset.from_dict({"id": [1, 2]})}) # Missing required cols
    with pytest.raises(DataLoadingError, match="Train dataset missing required columns"):
        select_and_limit_splits(bad_dict, pair_format_config)
