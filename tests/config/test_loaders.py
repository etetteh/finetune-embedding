# tests/config/test_loaders.py
import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Use absolute imports for testing
from finetune_embedding.config.loaders import (
    _create_arg_parser,
    _load_config_from_file,
    _structure_config_dict,
    load_and_validate_config,
)
from finetune_embedding.config.settings import (
    AppSettings,
    DatasetFormat,
)  # Import for type check
from finetune_embedding.exceptions import ConfigurationError

# --- Tests for _load_config_from_file (Error Path Testing) ---


def test_load_config_from_file_success(sample_json_file_factory):
    """Test loading a valid JSON config file."""
    config_content = {"model": {"model_name_or_path": "test-model"}}
    # Use the factory correctly
    config_path = sample_json_file_factory(content=config_content)
    loaded_config = _load_config_from_file(str(config_path))
    assert loaded_config == config_content


def test_load_config_from_file_not_found():
    """Test loading a non-existent file."""
    with pytest.raises(FileNotFoundError):
        _load_config_from_file("non_existent_file.json")


def test_load_config_from_file_invalid_json(tmp_path: Path):
    """Test loading an invalid JSON file."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{invalid json")
    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        _load_config_from_file(str(config_path))


# --- Tests for _create_arg_parser (Basic Check) ---
def test_create_arg_parser_returns_parser():
    parser = _create_arg_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Check if a few key arguments exist
    assert any(action.dest == "model_name_or_path" for action in parser._actions)
    assert any(action.dest == "output_dir" for action in parser._actions)
    assert any(action.dest == "use_lora" for action in parser._actions)


# --- Tests for _structure_config_dict (Parameterized & Basic) ---
@pytest.mark.parametrize(
    "flat_config, expected_model_path, expected_epochs",
    [
        ({"model_name_or_path": "model_a", "epochs": 3}, "model_a", 3),
        ({"model_name_or_path": "model_b", "epochs": 1}, "model_b", 1),
        ({"model_name_or_path": "model_c"}, "model_c", None),  # Epochs missing
    ],
    ids=["epochs_3", "epochs_1", "epochs_missing"],  # Test IDs for clarity
)
def test_structure_config_dict_basic(flat_config, expected_model_path, expected_epochs):
    """Test basic structuring of flat dict using parameterization."""
    # Ensure required keys are present for the structuring logic to work
    flat_config.setdefault("dataset_format", "pair")
    flat_config.setdefault("output_dir", "dummy")

    structured = _structure_config_dict(flat_config)
    assert structured["model"]["model_name_or_path"] == expected_model_path
    if expected_epochs is not None:
        assert structured["training"]["epochs"] == expected_epochs
    else:
        assert "epochs" not in structured.get("training", {})


def test_structure_config_dict_prefixed_args():
    """Test structuring with prefixed args like hnm_ and lora_."""
    flat_config = {
        "hnm_num_negatives": 5,
        "hnm_use_faiss": True,
        "lora_rank": 8,
        "use_lora": True,  # Note: use_lora is not prefixed
        "model_name_or_path": "dummy",
        "dataset_format": "pair",
        "output_dir": "dummy",
    }
    structured = _structure_config_dict(flat_config)
    assert structured["hnm"]["num_negatives"] == 5
    assert structured["hnm"]["use_faiss"] is True
    assert structured["lora"]["rank"] == 8
    assert structured["lora"]["use_lora"] is True


# --- Tests for load_and_validate_config (Mocking, Parameterized, Error Path) ---


@pytest.fixture(autouse=True)
def auto_mock_argparse(mocker):
    """Automatically mocks argparse methods used by load_and_validate_config."""
    mocks = {
        "parse_known_args": mocker.patch("argparse.ArgumentParser.parse_known_args"),
        "parse_args": mocker.patch("argparse.ArgumentParser.parse_args"),
        "set_defaults": mocker.patch("argparse.ArgumentParser.set_defaults"),
        "error": mocker.patch("argparse.ArgumentParser.error", side_effect=SystemExit),
    }
    yield mocks


@pytest.fixture(autouse=True)
def auto_mock_config_helpers(mocker):
    """Automatically mocks file loading and dotenv."""
    mocks = {
        "load_file": mocker.patch(
            "finetune_embedding.config.loaders._load_config_from_file"
        ),
        "load_dotenv": mocker.patch("finetune_embedding.config.loaders.load_dotenv"),
    }
    yield mocks


def test_load_validate_cli_only_minimal(
    auto_mock_argparse, auto_mock_config_helpers, mock_args_namespace
):
    """Test loading minimal required args via CLI only."""
    mock_args_namespace.config_file = None
    # Ensure the mock args being validated have a source
    mock_args_namespace.dataset_name = "mock-hub-dataset"

    auto_mock_argparse["parse_known_args"].return_value = (mock_args_namespace, [])
    auto_mock_argparse["parse_args"].return_value = mock_args_namespace

    settings = load_and_validate_config()

    assert isinstance(settings, AppSettings)
    assert settings.model.model_name_or_path == mock_args_namespace.model_name_or_path
    # Add assertion for dataset name
    assert settings.dataset.dataset_name == "mock-hub-dataset"
    assert str(settings.training.output_dir) == mock_args_namespace.output_dir
    auto_mock_config_helpers["load_file"].assert_not_called()
    auto_mock_config_helpers["load_dotenv"].assert_called_once()


def test_load_validate_config_file_override(
    auto_mock_argparse,
    auto_mock_config_helpers,
    sample_json_file_factory,
    minimal_config_dict,
    mock_args_namespace,
):
    """Test config file values overriding defaults."""
    config_content = minimal_config_dict.copy()
    config_content["training"]["epochs"] = 5
    config_content["model"]["model_name_or_path"] = "config-model"
    # Ensure dataset source is in the minimal dict being copied
    config_content["dataset"]["dataset_name"] = "config-hub-dataset"
    config_path = sample_json_file_factory(content=config_content)

    # Mock args indicating config file usage
    mock_args_namespace.config_file = str(config_path)
    mock_args_namespace.model_name_or_path = "config-model"  # Reflect config override
    mock_args_namespace.epochs = 5  # Reflect config override
    mock_args_namespace.dataset_name = "config-hub-dataset"  # Reflect config override

    mock_known_args = MagicMock(spec=argparse.Namespace, config_file=str(config_path))
    auto_mock_argparse["parse_known_args"].return_value = (
        mock_known_args,
        [],
    )  # Use the mock dict
    auto_mock_argparse["parse_args"].return_value = mock_args_namespace
    auto_mock_config_helpers["load_file"].return_value = config_content

    settings = load_and_validate_config()

    auto_mock_config_helpers["load_file"].assert_called_once_with(str(config_path))
    assert settings.model.model_name_or_path == "config-model"
    assert settings.training.epochs == 5
    assert settings.dataset.dataset_name == "config-hub-dataset"


def test_load_validate_cli_overrides_config(
    auto_mock_argparse,
    auto_mock_config_helpers,
    sample_json_file_factory,
    minimal_config_dict,
    mock_args_namespace,
):
    """Test CLI args overriding config file values."""
    config_content = minimal_config_dict.copy()
    config_content["training"]["epochs"] = 5
    config_content["model"]["model_name_or_path"] = "config-model"
    config_content["dataset"]["dataset_name"] = "config-hub-dataset"
    config_path = sample_json_file_factory(content=config_content)

    # Mock known args finding the config file
    mock_known_args = MagicMock(spec=argparse.Namespace, config_file=str(config_path))
    # CLI args provided *after* config file path
    remaining_cli = ["--epochs", "2", "--model_name_or_path", "cli-override-model"]
    auto_mock_argparse["parse_known_args"].return_value = (
        mock_known_args,
        remaining_cli,
    )  # Use the mock dict

    # Mock final args reflecting CLI overrides
    mock_args_namespace.config_file = str(config_path)
    mock_args_namespace.model_name_or_path = "cli-override-model"  # CLI override
    mock_args_namespace.epochs = 2  # CLI override
    mock_args_namespace.dataset_name = (
        "config-hub-dataset"  # From config (not overridden by CLI)
    )

    auto_mock_argparse["parse_args"].return_value = mock_args_namespace
    auto_mock_config_helpers["load_file"].return_value = config_content

    settings = load_and_validate_config()

    auto_mock_config_helpers["load_file"].assert_called_once_with(str(config_path))
    assert settings.model.model_name_or_path == "cli-override-model"
    assert settings.training.epochs == 2
    assert settings.dataset.dataset_name == "config-hub-dataset"


@pytest.mark.parametrize(
    "missing_arg", ["model_name_or_path", "dataset_format", "output_dir"]
)
def test_load_validate_missing_required_arg_fails(
    auto_mock_argparse, auto_mock_config_helpers, mock_args_namespace, missing_arg
):
    """Test failure when a required arg is missing (parameterized)."""
    mock_args_namespace.config_file = None
    # Ensure dataset source is present before removing another required field
    mock_args_namespace.dataset_name = "mock-hub-dataset"
    setattr(mock_args_namespace, missing_arg, None)  # Remove the required arg

    auto_mock_argparse["parse_known_args"].return_value = (mock_args_namespace, [])
    auto_mock_argparse["parse_args"].return_value = mock_args_namespace

    # Simply check that ConfigurationError is raised, as Pydantic validation fails first
    with pytest.raises(ConfigurationError):
        load_and_validate_config()


# --- Hypothesis Property-Based Test Example ---
# Strategy for generating valid dataset formats

dataset_format_strategy = st.sampled_from(list(DatasetFormat.__args__))


@given(ds_format=dataset_format_strategy)
@settings(
    max_examples=len(DatasetFormat.__args__),
    # --- ADD THIS LINE ---
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    # --- END ADD ---
)
# Keep mocker in the signature, as mocks are applied inside
def test_load_validate_dataset_format_property(mocker, mock_args_namespace, ds_format):
    """Test that valid dataset formats from CLI are correctly parsed."""
    # --- Setup mocks INSIDE the test function ---
    mock_parse_known_args = mocker.patch("argparse.ArgumentParser.parse_known_args")
    mock_parse_args = mocker.patch("argparse.ArgumentParser.parse_args")
    mocker.patch("argparse.ArgumentParser.set_defaults")  # Still need to patch this
    mock_load_file = mocker.patch(
        "finetune_embedding.config.loaders._load_config_from_file"
    )
    mocker.patch("finetune_embedding.config.loaders.load_dotenv")  # Mock dotenv load
    # --- End mock setup ---

    mock_args_namespace.config_file = None
    mock_args_namespace.dataset_format = ds_format  # Set format from hypothesis
    # Ensure dataset source is present
    mock_args_namespace.dataset_name = "mock-hub-dataset"

    # Configure mock return values for this specific example run
    mock_parse_known_args.return_value = (mock_args_namespace, [])
    mock_parse_args.return_value = mock_args_namespace

    settings = load_and_validate_config()

    assert settings.dataset.dataset_format == ds_format
    mock_load_file.assert_not_called()  # Verify file wasn't loaded
