# tests/test_main.py
import pytest
from unittest.mock import patch, MagicMock
import sys

# Use absolute imports
from finetune_embedding.main import main
from finetune_embedding.exceptions import ConfigurationError, FineTuningError

# Use fixtures: basic_app_settings

@patch('finetune_embedding.main.load_and_validate_config')
@patch('finetune_embedding.main.setup_logging')
@patch('finetune_embedding.main.FineTuningService')
def test_main_success_path(mock_service_cls, mock_setup_logging, mock_load_config, basic_app_settings):
    """Test the main function runs successfully."""
    mock_load_config.return_value = basic_app_settings
    mock_service_instance = mock_service_cls.return_value
    mock_service_instance.run_pipeline = MagicMock()

    # Patch sys.argv if load_and_validate_config relies on it directly
    # (Our current loader uses argparse internally, so mocking load_and_validate_config is enough)
    # with patch.object(sys, 'argv', ['main.py', '--some_arg']):
    exit_code = main()

    mock_load_config.assert_called_once()
    mock_setup_logging.assert_called_once_with(basic_app_settings.log_level, basic_app_settings.log_file)
    mock_service_cls.assert_called_once_with(basic_app_settings)
    mock_service_instance.run_pipeline.assert_called_once()
    assert exit_code == 0

@patch('finetune_embedding.main.load_and_validate_config', side_effect=ConfigurationError("Bad config"))
@patch('finetune_embedding.main.setup_logging')
@patch('finetune_embedding.main.FineTuningService')
def test_main_config_error(mock_service_cls, mock_setup_logging, mock_load_config, caplog):
    """Test main function handles ConfigurationError."""
    exit_code = main()

    mock_load_config.assert_called_once()
    mock_setup_logging.assert_not_called() # Logging setup happens after config load
    mock_service_cls.assert_not_called()
    assert exit_code == 1
    assert "Execution failed: Bad config" in caplog.text

@patch('finetune_embedding.main.load_and_validate_config')
@patch('finetune_embedding.main.setup_logging')
@patch('finetune_embedding.main.FineTuningService')
def test_main_pipeline_error(mock_service_cls, mock_setup_logging, mock_load_config, basic_app_settings, caplog):
    """Test main function handles errors from the pipeline service."""
    mock_load_config.return_value = basic_app_settings
    mock_service_instance = mock_service_cls.return_value
    mock_service_instance.run_pipeline.side_effect = FineTuningError("Pipeline crashed")

    exit_code = main()

    mock_load_config.assert_called_once()
    mock_setup_logging.assert_called_once()
    mock_service_cls.assert_called_once()
    mock_service_instance.run_pipeline.assert_called_once()
    assert exit_code == 1
    assert "Execution failed: Pipeline crashed" in caplog.text

@patch('finetune_embedding.main.load_and_validate_config')
@patch('finetune_embedding.main.setup_logging')
@patch('finetune_embedding.main.FineTuningService')
def test_main_unexpected_error(mock_service_cls, mock_setup_logging, mock_load_config, basic_app_settings, caplog):
    """Test main function handles unexpected errors."""
    mock_load_config.return_value = basic_app_settings
    mock_service_instance = mock_service_cls.return_value
    mock_service_instance.run_pipeline.side_effect = Exception("Something totally unexpected")

    exit_code = main()

    assert exit_code == 1
    assert "An unexpected critical error occurred: Something totally unexpected" in caplog.text

