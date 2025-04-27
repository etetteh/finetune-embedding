# tests/model/test_loaders.py
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch  # Import ANY

import pytest
import torch

from finetune_embedding.config.settings import ModelConfig, TrainingConfig
from finetune_embedding.exceptions import ModelError

# Use absolute imports
from finetune_embedding.model.loaders import initialize_model


# Fixtures for default configs
@pytest.fixture
def mock_model_config() -> ModelConfig:
    # Use a dummy name, it won't be loaded due to patching
    return ModelConfig(model_name_or_path="dummy/model-name")


@pytest.fixture
def mock_training_config(tmp_path: Path) -> TrainingConfig:
    # Provide a dummy output_dir
    return TrainingConfig(output_dir=tmp_path / "dummy_output")


# --- Tests ---


# --- FIX: Patch SentenceTransformer ---
@patch("finetune_embedding.model.loaders.SentenceTransformer")
def test_initialize_model_success(
    mock_st_class, mock_model_config, mock_training_config
):
    """Test successful model initialization (mocked)."""
    mock_model_instance = MagicMock()
    mock_model_instance.device = torch.device("cpu")  # Simulate device assignment
    mock_st_class.return_value = mock_model_instance  # Mock constructor return

    device = torch.device("cpu")
    model = initialize_model(mock_model_config, mock_training_config, device)

    # Assert SentenceTransformer was called with correct args
    mock_st_class.assert_called_once_with(
        model_name_or_path=mock_model_config.model_name_or_path,
        model_card_data=ANY,  # Check type if needed, but ANY is simpler
        trust_remote_code=mock_model_config.trust_remote_code,
        cache_folder=mock_model_config.cache_dir,
        model_kwargs={},  # No precision flags set in default training config
        device=str(device),
    )
    assert model == mock_model_instance  # Check returned object is the mock


# --- FIX: Patch SentenceTransformer ---
@pytest.mark.parametrize(
    "use_fp16, use_bf16, expected_dtype",
    [
        (True, False, torch.float16),
        (False, True, torch.bfloat16),
        (True, True, torch.bfloat16),  # Assuming BF16 priority
        (False, False, None),
    ],
    ids=["fp16", "bf16", "both", "none"],
)
@patch("finetune_embedding.model.loaders.SentenceTransformer")
def test_initialize_model_precision(
    mock_st_class,
    use_fp16,
    use_bf16,
    expected_dtype,
    mock_model_config,
    mock_training_config,
):
    """Test model initialization with different precision settings (mocked)."""
    mock_training_config.use_fp16 = use_fp16
    mock_training_config.use_bf16 = use_bf16

    mock_model_instance = MagicMock()
    mock_model_instance.device = torch.device("cpu")
    mock_st_class.return_value = mock_model_instance

    device = torch.device("cpu")
    model = initialize_model(mock_model_config, mock_training_config, device)

    expected_kwargs = {}
    if expected_dtype:
        expected_kwargs["torch_dtype"] = expected_dtype

    mock_st_class.assert_called_once_with(
        model_name_or_path=mock_model_config.model_name_or_path,
        model_card_data=ANY,
        trust_remote_code=mock_model_config.trust_remote_code,
        cache_folder=mock_model_config.cache_dir,
        model_kwargs=expected_kwargs,  # Check correct dtype is passed
        device=str(device),
    )
    assert model == mock_model_instance


# --- FIX: Patch SentenceTransformer where it's used ---
@patch(
    "finetune_embedding.model.loaders.SentenceTransformer",
    side_effect=OSError("Simulated loading error"),
)
def test_initialize_model_os_error(
    mock_st_class, mock_model_config, mock_training_config
):
    """Test handling of OSError during model loading."""
    with pytest.raises(
        ModelError, match="Failed to load model: Simulated loading error"
    ):
        initialize_model(mock_model_config, mock_training_config, torch.device("cpu"))
    mock_st_class.assert_called_once()  # Verify constructor was attempted


# --- FIX: Patch SentenceTransformer where it's used and adjust match ---
@patch(
    "finetune_embedding.model.loaders.SentenceTransformer",
    side_effect=Exception("Some other error"),
)
def test_initialize_model_other_error(
    mock_st_class, mock_model_config, mock_training_config
):
    """Test handling of generic exceptions during model loading."""
    # Expect the message from the generic except block
    with pytest.raises(
        ModelError, match="Model initialization failed: Some other error"
    ):
        initialize_model(mock_model_config, mock_training_config, torch.device("cpu"))
    mock_st_class.assert_called_once()  # Verify constructor was attempted
