# tests/data/test_hnm.py
import pytest
from unittest.mock import MagicMock, patch # Import patch
from datasets import Dataset
from sentence_transformers import SentenceTransformer # Import for spec

# Use absolute imports
# Import get_expected_columns if it's used in hnm.py
from finetune_embedding.data.preprocessing import get_expected_columns
from finetune_embedding.data.hnm import apply_hard_negative_mining
from finetune_embedding.data.loaders import DatasetContainer
from finetune_embedding.config.settings import HNMConfig, DatasetConfig
from finetune_embedding.exceptions import FineTuningError, DataLoadingError, ConfigurationError

# Use fixtures from conftest.py:
# mock_sentence_transformer_instance

@pytest.fixture
def pair_dataset_container() -> DatasetContainer:
    train_ds = Dataset.from_dict({"sentence1": ["a1", "a2"], "sentence2": ["p1", "p2"]})
    eval_ds = Dataset.from_dict({"sentence1": ["e1"], "sentence2": ["ep1"]})
    return DatasetContainer(train=train_ds, eval=eval_ds)

@pytest.fixture
def triplet_dataset_container() -> DatasetContainer:
    train_ds = Dataset.from_dict({"anchor": ["a1"], "positive": ["p1"], "negative": ["n1"]})
    return DatasetContainer(train=train_ds)

@pytest.fixture
def pair_dataset_config() -> DatasetConfig:
    # Ensure a dataset_name is present for get_expected_columns logic if needed
    return DatasetConfig(dataset_format="pair", dataset_name="mock_ds")

@pytest.fixture
def triplet_dataset_config() -> DatasetConfig:
    return DatasetConfig(dataset_format="triplet", dataset_name="mock_ds")

@pytest.fixture
def default_hnm_config() -> HNMConfig:
    return HNMConfig()

# This test PASSED before, should still pass
def test_hnm_skips_non_pair_format(triplet_dataset_container, mock_sentence_transformer_instance, default_hnm_config, triplet_dataset_config):
    """Test that HNM is skipped if the initial format is not 'pair'."""
    original_container = triplet_dataset_container
    # Mock locally just in case, though it shouldn't be called
    with patch('finetune_embedding.data.hnm.mine_hard_negatives') as mock_miner:
        result_container, effective_format = apply_hard_negative_mining(
            datasets=original_container,
            model=mock_sentence_transformer_instance,
            hnm_config=default_hnm_config,
            train_batch_size=16,
            dataset_config=triplet_dataset_config
        )
        mock_miner.assert_not_called() # Verify it wasn't called
    assert result_container == original_container # Should return original
    assert effective_format == "triplet" # Format should not change

# Use mocker.patch locally, use mock_sentence_transformer_instance fixture
def test_hnm_runs_for_pair_format(mocker, pair_dataset_container, mock_sentence_transformer_instance, default_hnm_config, pair_dataset_config):
    """Test that HNM runs and updates format for 'pair' datasets."""

    # --- Mock mine_hard_negatives locally ---
    # Define what the mock should return for train and eval datasets
    mock_train_result = pair_dataset_container.train_dataset.add_column("negative", ["neg_train"] * len(pair_dataset_container.train_dataset))
    mock_eval_result = pair_dataset_container.eval_dataset.add_column("negative", ["neg_eval"] * len(pair_dataset_container.eval_dataset))

    # Patch the function where it's used in your hnm module
    mock_miner = mocker.patch(
        'finetune_embedding.data.hnm.mine_hard_negatives',
        # Return different results based on which dataset is passed
        side_effect=[mock_train_result, mock_eval_result]
    )
    # --- End Mock Setup ---

    result_container, effective_format = apply_hard_negative_mining(
        datasets=pair_dataset_container,
        model=mock_sentence_transformer_instance, # Use the fixture again
        hnm_config=default_hnm_config,
        train_batch_size=16,
        dataset_config=pair_dataset_config
    )

    # Check the mock was called correctly
    assert mock_miner.call_count == 2
    # Check first call (train) args
    call1_args, call1_kwargs = mock_miner.call_args_list[0]
    assert call1_kwargs['dataset'] == pair_dataset_container.train_dataset
    assert call1_kwargs['model'] == mock_sentence_transformer_instance
    assert call1_kwargs['output_format'] == 'triplet'
    # Check second call (eval) args
    call2_args, call2_kwargs = mock_miner.call_args_list[1]
    assert call2_kwargs['dataset'] == pair_dataset_container.eval_dataset
    assert call2_kwargs['model'] == mock_sentence_transformer_instance

    # Check format updated
    assert effective_format == "triplet"
    # Check datasets returned are the ones from the mock
    assert result_container.train_dataset == mock_train_result
    assert result_container.eval_dataset == mock_eval_result
    assert "negative" in result_container.train_dataset.column_names
    assert "negative" in result_container.eval_dataset.column_names
    assert result_container.test_dataset is None # Test was None initially

# This test PASSED before, should still pass
def test_hnm_error_if_no_model(pair_dataset_container, default_hnm_config, pair_dataset_config):
    """Test error if model is None."""
    with pytest.raises(FineTuningError, match="Model must be initialized"):
        apply_hard_negative_mining(
            datasets=pair_dataset_container, model=None, hnm_config=default_hnm_config,
            train_batch_size=16, dataset_config=pair_dataset_config
        )

# This test PASSED before, should still pass
# Expect DataLoadingError because model check should pass now
def test_hnm_error_if_no_train_data(mock_sentence_transformer_instance, default_hnm_config, pair_dataset_config):
    """Test error if train dataset is missing."""
    empty_container = DatasetContainer(train=None)
    with pytest.raises(DataLoadingError, match="Training data must be loaded"):
        apply_hard_negative_mining(
            datasets=empty_container, model=mock_sentence_transformer_instance, hnm_config=default_hnm_config,
            train_batch_size=16, dataset_config=pair_dataset_config
        )

# This test PASSED before, should still pass
# Expect FineTuningError because model check passes, but mock raises error
def test_hnm_error_if_mining_fails(mocker, pair_dataset_container, mock_sentence_transformer_instance, default_hnm_config, pair_dataset_config):
    """Test error propagation if mine_hard_negatives fails."""
    # Mock locally to raise error
    mock_miner = mocker.patch(
        'finetune_embedding.data.hnm.mine_hard_negatives',
        side_effect=ValueError("Mining failed internally")
    )

    with pytest.raises(FineTuningError, match="Hard Negative Mining failed"):
        apply_hard_negative_mining(
            datasets=pair_dataset_container, model=mock_sentence_transformer_instance, hnm_config=default_hnm_config,
            train_batch_size=16, dataset_config=pair_dataset_config
        )
    # Verify the mock was called (once, for the train set, before failing)
    mock_miner.assert_called_once()
