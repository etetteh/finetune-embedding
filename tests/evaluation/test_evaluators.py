# tests/evaluation/test_evaluators.py
import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset
from sentence_transformers import evaluation

from finetune_embedding.evaluation.evaluators import create_evaluator
from finetune_embedding.config.settings import DatasetConfig
from finetune_embedding.exceptions import ConfigurationError

@pytest.fixture
def mock_dataset_config() -> DatasetConfig:
    """Provides a default DatasetConfig instance."""
    return DatasetConfig(dataset_format="pair", eval_split="eval", test_split="test") # Minimal valid

@pytest.fixture
def triplet_dataset() -> Dataset:
    return Dataset.from_dict({"anchor": ["a"], "positive": ["p"], "negative": ["n"]})

@pytest.fixture
def pair_score_dataset() -> Dataset:
    return Dataset.from_dict({"sentence1": ["s1"], "sentence2": ["s2"], "score": [0.8]})

@pytest.fixture
def pair_class_dataset() -> Dataset:
    # Include valid and invalid labels for filtering test
    return Dataset.from_dict({"sentence1": ["s1", "s3", "s5"], "sentence2": ["s2", "s4", "s6"], "label": [1, 0, 2]})

@patch('sentence_transformers.evaluation.TripletEvaluator')
def test_create_evaluator_triplet(mock_triplet_eval, triplet_dataset, mock_dataset_config):
    """Test creating TripletEvaluator."""
    mock_dataset_config.dataset_format = "triplet" # Ensure format matches
    evaluator = create_evaluator(triplet_dataset, "triplet", mock_dataset_config, 16, "eval")
    assert isinstance(evaluator, MagicMock) # Check if the mocked class instance was returned
    mock_triplet_eval.assert_called_once()
    call_kwargs = mock_triplet_eval.call_args.kwargs
    assert call_kwargs.get('name') == "eval_local-eval" # Check name generation
    assert call_kwargs.get('batch_size') == 16

@patch('sentence_transformers.evaluation.EmbeddingSimilarityEvaluator')
def test_create_evaluator_pair_score(mock_sim_eval, pair_score_dataset, mock_dataset_config):
    """Test creating EmbeddingSimilarityEvaluator."""
    mock_dataset_config.dataset_format = "pair-score"
    evaluator = create_evaluator(pair_score_dataset, "pair-score", mock_dataset_config, 16, "test")
    assert isinstance(evaluator, MagicMock)
    mock_sim_eval.assert_called_once()
    call_kwargs = mock_sim_eval.call_args.kwargs
    assert call_kwargs.get('name') == "test_local-test"
    assert call_kwargs.get('scores') == [0.8]

@patch('sentence_transformers.evaluation.BinaryClassificationEvaluator')
def test_create_evaluator_pair_class(mock_bin_eval, pair_class_dataset, mock_dataset_config):
    """Test creating BinaryClassificationEvaluator with filtering."""
    mock_dataset_config.dataset_format = "pair-class"
    evaluator = create_evaluator(pair_class_dataset, "pair-class", mock_dataset_config, 16, "eval")
    assert isinstance(evaluator, MagicMock)
    mock_bin_eval.assert_called_once()
    call_kwargs = mock_bin_eval.call_args.kwargs
    assert call_kwargs.get('name') == "eval_local-eval"
    # Check that only labels 0 and 1 were passed
    assert call_kwargs.get('labels') == [1, 0]
    assert call_kwargs.get('sentences1') == ["s1", "s3"]
    assert call_kwargs.get('sentences2') == ["s2", "s4"]

def test_create_evaluator_no_dataset(mock_dataset_config):
    """Test returning None if dataset is None or empty."""
    assert create_evaluator(None, "triplet", mock_dataset_config, 16) is None
    empty_ds = Dataset.from_dict({"anchor": [], "positive": [], "negative": []})
    assert create_evaluator(empty_ds, "triplet", mock_dataset_config, 16) is None

def test_create_evaluator_missing_columns(triplet_dataset, mock_dataset_config):
    """Test error if required columns are missing."""
    mock_dataset_config.dataset_format = "pair-score" # Mismatch format and data
    with pytest.raises(ConfigurationError, match="missing required columns"):
        create_evaluator(triplet_dataset, "pair-score", mock_dataset_config, 16)

def test_create_evaluator_pair_class_no_binary_labels(mock_dataset_config):
    """Test returning None if pair-class data has no valid binary labels."""
    ds = Dataset.from_dict({"sentence1": ["s1"], "sentence2": ["s2"], "label": [2]})
    mock_dataset_config.dataset_format = "pair-class"
    evaluator = create_evaluator(ds, "pair-class", mock_dataset_config, 16)
    assert evaluator is None

