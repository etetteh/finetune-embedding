# tests/training/test_trainer.py

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

# Assuming your project structure allows these imports
from finetune_embedding.config.settings import TrainingConfig
from finetune_embedding.training.trainer import TrainingWrapper
from finetune_embedding.exceptions import TrainingError, ConfigurationError

# Mock necessary classes from sentence_transformers and datasets
# You might need to adjust paths if your mocks are structured differently
@pytest.fixture
def mock_model():
    """Fixture for a mock SentenceTransformer model."""
    model = MagicMock()
    model.device = "cpu" # Mock device attribute
    return model

@pytest.fixture
def mock_loss():
    """Fixture for a mock loss function."""
    return MagicMock()

@pytest.fixture
def mock_dataset():
    """Fixture for a mock datasets.Dataset."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100 # Give it a length
    return dataset

@pytest.fixture
def mock_evaluator():
    """Fixture for a mock SentenceEvaluator."""
    evaluator = MagicMock()
    # Mock the specific evaluator types for isinstance checks
    # Use patch to temporarily change the class for isinstance
    with patch('finetune_embedding.training.trainer.TripletEvaluator', MagicMock), \
         patch('finetune_embedding.training.trainer.EmbeddingSimilarityEvaluator', MagicMock), \
         patch('finetune_embedding.training.trainer.BinaryClassificationEvaluator', MagicMock):
        # Make the mock evaluator an instance of a specific (mocked) type if needed for tests
        # For example, make it a BinaryClassificationEvaluator for metric inference tests
        evaluator.__class__ = patch('finetune_embedding.training.trainer.BinaryClassificationEvaluator').__enter__()
        yield evaluator # Yield the configured mock

@pytest.fixture
def minimal_training_config(tmp_path):
    """Provides a minimal valid TrainingConfig."""
    return TrainingConfig(
        output_dir=tmp_path / "test_output",
        epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=1e-5,
        # Add other required fields with defaults if necessary
        # Default save_strategy is 'steps'
    )

# --- Tests for _create_training_args ---

def test_create_training_args_basic(minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test basic argument creation without evaluation."""
    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
        # No eval_dataset or dev_evaluator
    )
    args = wrapper._create_training_args()

    assert args.output_dir == str(minimal_training_config.output_dir)
    assert args.num_train_epochs == minimal_training_config.epochs
    assert args.per_device_train_batch_size == minimal_training_config.train_batch_size
    assert args.learning_rate == minimal_training_config.learning_rate
    assert args.eval_strategy == "no" # Should default to 'no' if no evaluator
    assert args.load_best_model_at_end is False
    assert args.metric_for_best_model is None
    assert os.path.exists(minimal_training_config.output_dir) # Check dir creation

def test_create_training_args_with_evaluator(minimal_training_config, mock_model, mock_loss, mock_dataset, mock_evaluator):
    """Test argument creation with an evaluator and default metric inference."""
    minimal_training_config.eval_strategy = "steps" # Enable evaluation
    # save_strategy defaults to "steps", so it matches eval_strategy here
    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset, # Provide eval dataset
        dev_evaluator=mock_evaluator, # Provide evaluator
        training_args_config=minimal_training_config,
        dataset_name="my_data",
        eval_split_name="dev"
    )
    args = wrapper._create_training_args()

    assert args.eval_strategy == "steps"
    assert args.load_best_model_at_end is True
    # Check if the metric was inferred correctly (depends on mock_evaluator's mocked class)
    assert args.metric_for_best_model == "eval_my_data-dev_cosine_accuracy" # Assuming BinaryClassificationEvaluator mock
    assert args.greater_is_better is True

def test_create_training_args_with_evaluator_user_metric(minimal_training_config, mock_model, mock_loss, mock_dataset, mock_evaluator):
    """Test argument creation with an evaluator and user-specified metric."""
    minimal_training_config.eval_strategy = "epoch"
    minimal_training_config.metric_for_best_model = "eval_custom_metric"
    # --- FIX: Ensure save_strategy matches eval_strategy when load_best_model_at_end=True ---
    minimal_training_config.save_strategy = "epoch"
    # --- END FIX ---

    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
        dev_evaluator=mock_evaluator,
        training_args_config=minimal_training_config,
    )
    args = wrapper._create_training_args()

    assert args.eval_strategy == "epoch"
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_custom_metric" # User metric overrides inference
    assert args.greater_is_better is True # Default assumption

def test_create_training_args_eval_strategy_override(minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test that eval_strategy is set to 'no' if specified but no evaluator is provided."""
    minimal_training_config.eval_strategy = "steps" # User wants eval
    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
        # No evaluator provided
    )
    args = wrapper._create_training_args()

    assert args.eval_strategy == "no" # Should be overridden
    assert args.load_best_model_at_end is False
    assert args.metric_for_best_model is None

def test_create_training_args_missing_output_dir(minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test error if output_dir is missing."""
    minimal_training_config.output_dir = None
    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
    )
    with pytest.raises(ConfigurationError, match="output_dir is required"):
        wrapper._create_training_args()

def test_create_training_args_dir_creation_fails(minimal_training_config, mock_model, mock_loss, mock_dataset, mocker):
    """Test error if output directory creation fails."""
    mock_makedirs = mocker.patch("os.makedirs", side_effect=OSError("Permission denied"))
    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
    )
    with pytest.raises(ConfigurationError, match="Failed to create output directory"):
        wrapper._create_training_args()
    mock_makedirs.assert_called_once_with(minimal_training_config.output_dir, exist_ok=True)


# --- Tests for train ---

@patch("finetune_embedding.training.trainer.SentenceTransformerTrainer")
def test_training_wrapper_train_success(mock_sbert_trainer, minimal_training_config, mock_model, mock_loss, mock_dataset, mock_evaluator):
    """Test successful training execution."""
    minimal_training_config.eval_strategy = "steps"
    # save_strategy defaults to "steps", matching eval_strategy
    mock_trainer_instance = mock_sbert_trainer.return_value
    mock_trainer_instance.train.return_value = MagicMock() # Mock train result

    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
        dev_evaluator=mock_evaluator,
        training_args_config=minimal_training_config,
    )

    wrapper.train()

    # Check that SentenceTransformerTrainer was initialized correctly
    mock_sbert_trainer.assert_called_once_with(
        model=mock_model,
        args=ANY, # Check args separately if needed, but _create_training_args tests cover it
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset, # Should be passed if evaluator exists
        loss=mock_loss,
        evaluator=mock_evaluator,
    )
    # Check that the trainer's train method was called
    mock_trainer_instance.train.assert_called_once()
    assert wrapper.trainer is mock_trainer_instance

@patch("finetune_embedding.training.trainer.SentenceTransformerTrainer")
def test_training_wrapper_train_handles_no_evaluator(mock_sbert_trainer, minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test training runs without an evaluator."""
    mock_trainer_instance = mock_sbert_trainer.return_value
    mock_trainer_instance.train.return_value = MagicMock()

    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
        # No eval_dataset or dev_evaluator
    )

    wrapper.train()

    # Check that eval_dataset is None when initializing trainer
    mock_sbert_trainer.assert_called_once_with(
        model=mock_model,
        args=ANY,
        train_dataset=mock_dataset,
        eval_dataset=None, # Correctly None
        loss=mock_loss,
        evaluator=None, # Correctly None
    )
    mock_trainer_instance.train.assert_called_once()

def test_training_wrapper_train_error_propagation(minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test errors during prerequisite checks."""
    # No model
    wrapper_no_model = TrainingWrapper(model=None, loss=mock_loss, train_dataset=mock_dataset, training_args_config=minimal_training_config)
    with pytest.raises(TrainingError, match="Model not initialized"):
        wrapper_no_model.train()

    # No train dataset
    wrapper_no_data = TrainingWrapper(model=mock_model, loss=mock_loss, train_dataset=None, training_args_config=minimal_training_config)
    with pytest.raises(TrainingError, match="Train dataset not loaded"):
        wrapper_no_data.train()

    # No loss
    wrapper_no_loss = TrainingWrapper(model=mock_model, loss=None, train_dataset=mock_dataset, training_args_config=minimal_training_config)
    with pytest.raises(TrainingError, match="Loss function not created"):
        wrapper_no_loss.train()

@patch("finetune_embedding.training.trainer.SentenceTransformerTrainer")
def test_training_wrapper_train_execution_error(mock_sbert_trainer, minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test handling of exceptions raised by trainer.train()."""
    mock_trainer_instance = mock_sbert_trainer.return_value
    mock_trainer_instance.train.side_effect = RuntimeError("CUDA out of memory")

    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
    )

    with pytest.raises(TrainingError, match="Training execution failed."):
        wrapper.train()

    mock_trainer_instance.train.assert_called_once()


# --- Tests for save_final_model ---

def test_save_final_model_success(minimal_training_config, mock_model, mock_loss, mock_dataset, mocker):
    """Test successful saving of the final model."""
    # Mock the trainer instance that would be created during train()
    mock_trainer_instance = MagicMock()
    mock_save = mock_trainer_instance.save_model

    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
    )
    # Manually set the trainer and args as if train() was called
    wrapper.trainer = mock_trainer_instance
    wrapper.training_args = wrapper._create_training_args() # Ensure args are created

    expected_save_path = os.path.join(str(minimal_training_config.output_dir), "final_model")

    saved_path = wrapper.save_final_model()

    mock_save.assert_called_once_with(expected_save_path)
    assert saved_path == expected_save_path

def test_save_final_model_no_trainer(minimal_training_config, mock_model, mock_loss, mock_dataset):
    """Test error when trying to save without a trainer instance."""
    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
    )
    # train() was not called, so wrapper.trainer is None
    with pytest.raises(TrainingError, match="Trainer not initialized"):
        wrapper.save_final_model()

def test_save_final_model_save_exception(minimal_training_config, mock_model, mock_loss, mock_dataset, mocker):
    """Test error handling when trainer.save_model fails."""
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.save_model.side_effect = OSError("Disk full")

    wrapper = TrainingWrapper(
        model=mock_model,
        loss=mock_loss,
        train_dataset=mock_dataset,
        training_args_config=minimal_training_config,
    )
    wrapper.trainer = mock_trainer_instance
    wrapper.training_args = wrapper._create_training_args()

    with pytest.raises(TrainingError, match="Failed to save final model"):
        wrapper.save_final_model()

    expected_save_path = os.path.join(str(minimal_training_config.output_dir), "final_model")
    mock_trainer_instance.save_model.assert_called_once_with(expected_save_path)

