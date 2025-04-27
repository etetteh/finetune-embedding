# /Users/en_tetteh/Documents/GitHub/finetune-embedding/tests/model/test_lora.py
from unittest.mock import MagicMock

import pytest

# Corrected import: Use LoRAConfig instead of LoraSettings
from finetune_embedding.config.settings import LoRAConfig

# Import exceptions raised by the function
from finetune_embedding.exceptions import ConfigurationError, ModelError

# Assuming your function is here:
from finetune_embedding.model.lora import add_lora_adapter


# Mock SentenceTransformer for testing purposes
class MockSentenceTransformer:
    def __init__(self):
        # Mock the underlying transformer model access
        self.auto_model = MagicMock(
            name="auto_model_instance"
        )  # Give it a name for clarity
        # Mock the add_adapter method
        self.add_adapter = MagicMock(name="add_adapter_method")

        # Create and hold a consistent mock module
        self._mock_module_instance = MagicMock(name="mock_module")
        self._mock_module_instance.auto_model = self.auto_model

    def _first_module(self):
        # Helper to simulate accessing the first module containing auto_model
        return self._mock_module_instance


@pytest.fixture
def mock_model() -> MockSentenceTransformer:
    """Provides a mock SentenceTransformer instance."""
    return MockSentenceTransformer()


@pytest.fixture
def valid_lora_settings() -> LoRAConfig:
    """Provides a valid LoRAConfig configuration."""
    # Assuming default rank/alpha/dropout are valid per Pydantic model
    return LoRAConfig(
        use_lora=True,
        rank=8,
        alpha=16,
        dropout=0.1,
        target_modules=["query", "key", "value"],
    )


@pytest.fixture
def disabled_lora_settings() -> LoRAConfig:
    """Provides LoRAConfig with use_lora=False."""
    return LoRAConfig(use_lora=False)


# --- Test Cases ---


def test_add_lora_adapter_success(mocker, mock_model, valid_lora_settings):
    """Test successful addition of LoRA adapter."""
    # Patch the original peft classes
    mock_lora_config_cls = mocker.patch("peft.LoraConfig")
    mock_task_type = mocker.patch("peft.TaskType")
    mocker.patch("peft.PeftModel")  # Patch PeftModel from peft

    # Ensure the mock model's underlying auto_model is not initially a PeftModel instance
    # Access it correctly via the mock module instance
    mock_model._first_module().auto_model.__class__ = (
        MagicMock  # Set class to a generic mock
    )

    add_lora_adapter(mock_model, valid_lora_settings)

    # Assert LoraConfig (from peft) was called with correct parameters from LoRAConfig (from settings)
    mock_lora_config_cls.assert_called_once_with(
        task_type=mock_task_type.FEATURE_EXTRACTION,  # Use the mocked TaskType
        r=valid_lora_settings.rank,
        lora_alpha=valid_lora_settings.alpha,
        lora_dropout=valid_lora_settings.dropout,
        target_modules=valid_lora_settings.target_modules,
        inference_mode=False,
    )

    # Assert model.add_adapter was called with the created config
    mock_model.add_adapter.assert_called_once_with(mock_lora_config_cls.return_value)


def test_add_lora_adapter_skipped(mocker, mock_model, disabled_lora_settings):
    """Test that adapter addition is skipped when use_lora is False."""
    # Patch the original peft class
    mock_lora_config_cls = mocker.patch("peft.LoraConfig")

    add_lora_adapter(mock_model, disabled_lora_settings)

    # Assert LoraConfig and add_adapter were NOT called
    mock_lora_config_cls.assert_not_called()
    mock_model.add_adapter.assert_not_called()


def test_add_lora_adapter_import_error(mocker, mock_model, valid_lora_settings):
    """Test handling of ImportError if peft is not installed."""
    # Patch the original peft classes to raise ImportError
    mocker.patch("peft.LoraConfig", side_effect=ImportError("No module named 'peft'"))
    mocker.patch("peft.TaskType", side_effect=ImportError("No module named 'peft'"))
    mocker.patch("peft.PeftModel", side_effect=ImportError("No module named 'peft'"))

    # Expect ConfigurationError because the except block catches ImportError
    with pytest.raises(ConfigurationError, match="PEFT library is required for LoRA."):
        add_lora_adapter(mock_model, valid_lora_settings)

    mock_model.add_adapter.assert_not_called()


def test_add_lora_adapter_peft_add_error(mocker, mock_model, valid_lora_settings):
    """Test handling of errors during model.add_adapter."""
    # Patch the original peft classes
    mocker.patch("peft.LoraConfig")
    mocker.patch("peft.TaskType")
    mocker.patch("peft.PeftModel")

    # Make the add_adapter call fail
    mock_model.add_adapter.side_effect = Exception("PEFT internal error")

    # Expect ModelError because the generic except block catches the Exception
    with pytest.raises(
        ModelError, match="Failed to add LoRA adapter: PEFT internal error"
    ):
        add_lora_adapter(mock_model, valid_lora_settings)

    mock_model.add_adapter.assert_called_once()  # Ensure it was called before failing


def test_add_lora_adapter_already_peft_warning(mocker, mock_model, valid_lora_settings):
    """Test that a warning is logged if the base model is already a PeftModel."""
    # Patch dependencies needed before the check
    mock_lora_config_cls = mocker.patch("peft.LoraConfig")
    mocker.patch("peft.TaskType")
    # Patch PeftModel as it's imported locally in the function
    mocker.patch("peft.PeftModel")
    mock_logger_warning = mocker.patch(
        "finetune_embedding.model.lora.logger.warning"
    )  # Patch the logger

    # --- FIX: Patch isinstance within the lora module ---
    # This forces the condition `isinstance(base_transformer_model, PeftModel)` to be True
    # for the purpose of this test, regardless of the actual class complexities.
    mocker.patch("finetune_embedding.model.lora.isinstance", return_value=True)

    # Call the function
    add_lora_adapter(mock_model, valid_lora_settings)

    # Assert the warning was logged
    mock_logger_warning.assert_called_once_with(
        "Base model already has PEFT config. Adding another LoRA adapter. Ensure this is intended."
    )

    # Assert adapter was still added (these should still be called after the warning)
    mock_lora_config_cls.assert_called_once()
    mock_model.add_adapter.assert_called_once()
