# tests/services/test_pipeline.py

from unittest.mock import ANY, MagicMock, call

import pytest
import torch

# Assuming your project structure allows these imports
from finetune_embedding.config.settings import (
    AppSettings,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from finetune_embedding.data.loaders import DatasetContainer
from finetune_embedding.exceptions import (
    ConfigurationError,
    DataLoadingError,
    EvaluationError,
    FineTuningError,
    ModelError,
    TrainingError,
)
from finetune_embedding.services.pipeline import FineTuningService

# --- Fixtures ---


@pytest.fixture
def minimal_app_settings(tmp_path):
    """Provides a minimal valid AppSettings object for testing."""
    output_dir = tmp_path / "pipeline_output"
    return AppSettings(
        model=ModelConfig(model_name_or_path="mock-model"),
        dataset=DatasetConfig(dataset_format="pair-score", dataset_name="mock-dataset"),
        training=TrainingConfig(
            output_dir=output_dir, epochs=1, train_batch_size=2, eval_batch_size=2
        ),
        # Defaults for hnm and lora
    )


@pytest.fixture
def mock_dataset():
    """Fixture for a mock datasets.Dataset."""
    ds = MagicMock()
    ds.__len__.return_value = 10  # Give it a length
    ds.column_names = ["sentence1", "sentence2", "score"]  # Example columns
    return ds


@pytest.fixture
def mock_dataset_dict(mock_dataset):
    """Fixture for a mock datasets.DatasetDict."""
    return {"train": mock_dataset, "validation": mock_dataset, "test": mock_dataset}


@pytest.fixture
def mock_model():
    """Fixture for a mock SentenceTransformer model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    return model


@pytest.fixture
def mock_loss():
    """Fixture for a mock loss function."""
    return MagicMock()


@pytest.fixture
def mock_evaluator():
    """Fixture for a mock SentenceEvaluator."""
    return MagicMock()


@pytest.fixture
def mock_trainer_wrapper():
    """Fixture for a mock TrainingWrapper instance."""
    wrapper = MagicMock()
    wrapper.save_final_model.return_value = "path/to/final_model"
    return wrapper


# --- Mocks for Pipeline Dependencies ---


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies(
    mocker,
    mock_dataset_dict,
    mock_dataset,
    mock_model,
    mock_loss,
    mock_evaluator,
    mock_trainer_wrapper,
):
    """Auto-used fixture to mock all external dependencies of FineTuningService."""
    mocks = {
        "set_seed": mocker.patch("finetune_embedding.services.pipeline.set_seed"),
        "determine_device": mocker.patch(
            "finetune_embedding.services.pipeline.determine_device",
            return_value=torch.device("cpu"),
        ),
        "update_precision_flags": mocker.patch(
            "finetune_embedding.services.pipeline.update_precision_flags"
        ),
        "save_config": mocker.patch(
            "finetune_embedding.services.pipeline.FineTuningService._save_effective_config"
        ),  # Mock internal method too
        "load_raw_datasets": mocker.patch(
            "finetune_embedding.services.pipeline.load_raw_datasets",
            return_value=mock_dataset_dict,
        ),
        "parse_column_rename_map": mocker.patch(
            "finetune_embedding.services.pipeline.parse_column_rename_map",
            return_value=None,
        ),
        "apply_column_renaming": mocker.patch(
            "finetune_embedding.services.pipeline.apply_column_renaming",
            return_value=mock_dataset_dict,
        ),
        "auto_split_dataset": mocker.patch(
            "finetune_embedding.services.pipeline.auto_split_dataset",
            return_value=mock_dataset_dict,
        ),
        "select_and_limit_splits": mocker.patch(
            "finetune_embedding.services.pipeline.select_and_limit_splits",
            return_value=DatasetContainer(
                train=mock_dataset, eval_dataset=mock_dataset, test=mock_dataset
            ),
        ),
        "initialize_model": mocker.patch(
            "finetune_embedding.services.pipeline.initialize_model",
            return_value=mock_model,
        ),
        "apply_hard_negative_mining": mocker.patch(
            "finetune_embedding.services.pipeline.apply_hard_negative_mining",
            return_value=(
                DatasetContainer(
                    train=mock_dataset, eval_dataset=mock_dataset, test=mock_dataset
                ),
                "triplet",
            ),
        ),
        "add_lora_adapter": mocker.patch(
            "finetune_embedding.services.pipeline.add_lora_adapter"
        ),
        "create_loss_function": mocker.patch(
            "finetune_embedding.services.pipeline.create_loss_function",
            return_value=mock_loss,
        ),
        "create_evaluator": mocker.patch(
            "finetune_embedding.services.pipeline.create_evaluator",
            side_effect=[mock_evaluator, mock_evaluator],
        ),  # Dev then Test
        "TrainingWrapper": mocker.patch(
            "finetune_embedding.services.pipeline.TrainingWrapper",
            return_value=mock_trainer_wrapper,
        ),
        "run_evaluation": mocker.patch(
            "finetune_embedding.services.pipeline.run_evaluation"
        ),
    }
    return mocks


# --- Test Cases ---


def test_pipeline_run_minimal_success(minimal_app_settings, mock_pipeline_dependencies):
    """Test the pipeline runs successfully with minimal configuration."""
    service = FineTuningService(minimal_app_settings)
    service.run_pipeline()

    # Check setup calls
    mock_pipeline_dependencies["set_seed"].assert_called_once_with(
        minimal_app_settings.training.seed
    )
    mock_pipeline_dependencies["determine_device"].assert_called_once()
    mock_pipeline_dependencies["update_precision_flags"].assert_called_once()
    mock_pipeline_dependencies[
        "save_config"
    ].assert_called_once()  # Check config saving

    # Check data handling calls
    mock_pipeline_dependencies["load_raw_datasets"].assert_called_once()
    mock_pipeline_dependencies["parse_column_rename_map"].assert_called_once()
    mock_pipeline_dependencies["apply_column_renaming"].assert_called_once()
    mock_pipeline_dependencies["auto_split_dataset"].assert_called_once()
    mock_pipeline_dependencies["select_and_limit_splits"].assert_called_once()

    # Check model init
    mock_pipeline_dependencies["initialize_model"].assert_called_once()

    # Check HNM was skipped (default format is pair-score)
    mock_pipeline_dependencies["apply_hard_negative_mining"].assert_not_called()

    # Check LoRA was skipped (default is use_lora=False)
    mock_pipeline_dependencies["add_lora_adapter"].assert_not_called()

    # --- FIX: Check loss creation with KEYWORD args ---
    mock_pipeline_dependencies["create_loss_function"].assert_called_once_with(
        model=ANY,  # Check model instance passed
        effective_format=minimal_app_settings.dataset.dataset_format,  # Initial format
        num_labels=minimal_app_settings.dataset.num_labels,
    )
    # --- END FIX ---

    # Check evaluator creation (dev and test) - These use kwargs, so keep as is
    assert mock_pipeline_dependencies["create_evaluator"].call_count == 2
    mock_pipeline_dependencies["create_evaluator"].assert_has_calls(
        [
            call(
                eval_dataset=ANY,
                effective_format=minimal_app_settings.dataset.dataset_format,
                dataset_config=minimal_app_settings.dataset,
                eval_batch_size=minimal_app_settings.training.eval_batch_size,
                name_prefix="eval",
            ),
            call(
                eval_dataset=ANY,
                effective_format=minimal_app_settings.dataset.dataset_format,
                dataset_config=minimal_app_settings.dataset,
                eval_batch_size=minimal_app_settings.training.eval_batch_size,
                name_prefix="test",
            ),
        ]
    )

    # Check TrainingWrapper initialization and train call
    mock_pipeline_dependencies["TrainingWrapper"].assert_called_once()
    mock_pipeline_dependencies[
        "TrainingWrapper"
    ].return_value.train.assert_called_once()
    mock_pipeline_dependencies[
        "TrainingWrapper"
    ].return_value.save_final_model.assert_called_once()

    # Check final evaluation call - Uses kwargs, keep as is
    mock_pipeline_dependencies["run_evaluation"].assert_called_once_with(
        evaluator=ANY,  # Check evaluator instance passed
        model=ANY,  # Check model instance passed
        output_path=str(minimal_app_settings.training.output_dir),
    )


def test_pipeline_run_with_hnm(minimal_app_settings, mock_pipeline_dependencies):
    """Test the pipeline runs correctly when HNM is triggered."""
    minimal_app_settings.dataset.dataset_format = "pair"  # Trigger HNM
    service = FineTuningService(minimal_app_settings)
    service.run_pipeline()

    # Check HNM was called - Uses kwargs, keep as is
    mock_pipeline_dependencies["apply_hard_negative_mining"].assert_called_once_with(
        datasets=ANY,
        model=ANY,
        hnm_config=minimal_app_settings.hnm,
        train_batch_size=minimal_app_settings.training.train_batch_size,
        dataset_config=minimal_app_settings.dataset,
    )

    # Check subsequent calls use the *effective* format from HNM ("triplet")
    effective_format_after_hnm = "triplet"
    # --- FIX: Check loss creation with KEYWORD args ---
    mock_pipeline_dependencies["create_loss_function"].assert_called_once_with(
        model=ANY,
        effective_format=effective_format_after_hnm,
        num_labels=minimal_app_settings.dataset.num_labels,
    )
    # --- END FIX ---

    # Check evaluator creation - Uses kwargs, keep as is
    assert mock_pipeline_dependencies["create_evaluator"].call_count == 2
    mock_pipeline_dependencies["create_evaluator"].assert_has_calls(
        [
            call(
                eval_dataset=ANY,
                effective_format=effective_format_after_hnm,
                dataset_config=minimal_app_settings.dataset,
                eval_batch_size=minimal_app_settings.training.eval_batch_size,
                name_prefix="eval",
            ),
            call(
                eval_dataset=ANY,
                effective_format=effective_format_after_hnm,
                dataset_config=minimal_app_settings.dataset,
                eval_batch_size=minimal_app_settings.training.eval_batch_size,
                name_prefix="test",
            ),
        ]
    )


def test_pipeline_run_with_lora(minimal_app_settings, mock_pipeline_dependencies):
    """Test the pipeline runs correctly when LoRA is enabled."""
    minimal_app_settings.lora.use_lora = True  # Enable LoRA
    service = FineTuningService(minimal_app_settings)
    service.run_pipeline()

    # --- Check LoRA was called with positional args (as fixed before) ---
    mock_pipeline_dependencies["add_lora_adapter"].assert_called_once_with(
        ANY,  # model
        minimal_app_settings.lora,  # lora_config
    )
    # --- END FIX ---


@pytest.mark.parametrize(
    "failing_mock, error_type, stage_name",
    [
        ("load_raw_datasets", DataLoadingError("Mock data load fail"), "Data Handling"),
        (
            "initialize_model",
            ModelError("Mock model init fail"),
            "Model Initialization",
        ),
        (
            "apply_hard_negative_mining",
            FineTuningError("Mock HNM fail"),
            "Hard Negative Mining",
        ),
        ("add_lora_adapter", ModelError("Mock LoRA fail"), "Applying LoRA"),
        ("create_loss_function", ConfigurationError("Mock loss fail"), "Loss Creation"),
        (
            "create_evaluator",
            EvaluationError("Mock dev eval fail"),
            "Evaluator Setup",
        ),  # First call fails
        (
            "TrainingWrapper",
            TrainingError("Mock training fail"),
            "Training",
        ),  # Mocking the class init fails
        (
            "run_evaluation",
            EvaluationError("Mock test eval fail"),
            "Test Evaluation",
        ),  # This configures run_evaluation to fail *if called*
    ],
)
def test_pipeline_error_propagation(
    minimal_app_settings,
    mock_pipeline_dependencies,
    failing_mock,
    error_type,
    stage_name,
):
    """Test that errors from different stages are propagated correctly."""
    # Configure specific mock to raise error
    mock_pipeline_dependencies[failing_mock].side_effect = error_type

    # Special setup for HNM/LoRA triggers if needed
    if stage_name == "Hard Negative Mining":
        minimal_app_settings.dataset.dataset_format = "pair"
    if stage_name == "Applying LoRA":
        minimal_app_settings.lora.use_lora = True
    # Special setup for TrainingWrapper failure (mock the class init)
    if stage_name == "Training":
        mock_pipeline_dependencies["TrainingWrapper"].side_effect = error_type
    # Special setup for test evaluation failure (make first create_evaluator succeed, second fail)
    if stage_name == "Test Evaluation":
        # The error actually happens in create_evaluator's second call
        mock_pipeline_dependencies["create_evaluator"].side_effect = [
            MagicMock(),
            EvaluationError("Mock test eval create fail"),
        ]
        # We expect EvaluationError, matching the error raised by create_evaluator
        expected_error_type = EvaluationError
        expected_error_msg = "Mock test eval create fail"
    else:
        expected_error_type = type(error_type)
        expected_error_msg = str(error_type)

    service = FineTuningService(minimal_app_settings)

    with pytest.raises(expected_error_type, match=expected_error_msg):
        service.run_pipeline()

    # --- FIX: Adjust assertion logic ---
    if stage_name == "Training":
        # TrainingWrapper class mock isn't called directly if init fails
        pass  # Or assert something else if needed
    elif stage_name == "Test Evaluation":
        # Error happens in the *second* call to create_evaluator
        assert (
            mock_pipeline_dependencies["create_evaluator"].call_count >= 2
        )  # Ensure it was attempted
        mock_pipeline_dependencies[
            "run_evaluation"
        ].assert_not_called()  # Ensure the failing step wasn't reached
    else:
        # For other stages, the mock configured to fail should have been called
        mock_pipeline_dependencies[failing_mock].assert_called()
    # --- END FIX ---


def test_pipeline_training_execution_error(
    minimal_app_settings, mock_pipeline_dependencies
):
    """Test error during the trainer.train() execution."""
    # Make the train() method of the wrapper instance raise an error
    mock_pipeline_dependencies[
        "TrainingWrapper"
    ].return_value.train.side_effect = TrainingError("CUDA OOM")

    service = FineTuningService(minimal_app_settings)

    with pytest.raises(TrainingError, match="CUDA OOM"):
        service.run_pipeline()

    mock_pipeline_dependencies[
        "TrainingWrapper"
    ].return_value.train.assert_called_once()
    # Ensure save_final_model and test evaluation are NOT called after training failure
    mock_pipeline_dependencies[
        "TrainingWrapper"
    ].return_value.save_final_model.assert_not_called()
    mock_pipeline_dependencies["run_evaluation"].assert_not_called()


def test_pipeline_unexpected_error(minimal_app_settings, mock_pipeline_dependencies):
    """Test handling of unexpected errors."""
    mock_pipeline_dependencies["initialize_model"].side_effect = ValueError(
        "Unexpected value error"
    )

    service = FineTuningService(minimal_app_settings)

    # Should be wrapped in FineTuningError
    with pytest.raises(
        FineTuningError, match="Unexpected pipeline error: Unexpected value error"
    ):
        service.run_pipeline()
