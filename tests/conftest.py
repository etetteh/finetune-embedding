# tests/conftest.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer  # Import for spec

# Import SentenceTransformerTrainingArguments for spec
from sentence_transformers.trainer import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

# --- Helper Functions for mock_args_namespace ---


def _flatten_nested_config(
    nested_dict: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    """Flattens a nested dictionary, adding prefixes for specific sections."""
    items: Dict[str, Any] = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        prefix = ""
        # Apply prefixes based on the top-level section key (parent_key)
        if parent_key == "hnm":
            prefix = "hnm_"
        elif parent_key == "lora":
            prefix = "lora_"

        # Construct the final key name for the flat dictionary
        flat_key = f"{prefix}{k}"

        if isinstance(v, dict):
            # Recursively flatten sub-dictionaries, passing the current key as parent
            items.update(_flatten_nested_config(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Handle list args directly
            items[flat_key] = v
        elif v is not None:
            # Add non-None scalar values
            items[flat_key] = v
        # We intentionally skip adding keys with None values here,
        # they will be handled by parser defaults later if needed.

    return items


def _add_explicit_boolean_flags(
    flat_args: Dict[str, Any], full_config_dict: Dict[str, Any]
):
    """Adds boolean flags explicitly based on the full config, handling defaults."""

    # Helper to safely get nested values, returning default if section is None or key missing
    def safe_get(section_name: str, key_name: str, default_value: bool) -> bool:
        section = full_config_dict.get(section_name)  # Get section, might be None
        if isinstance(section, dict):  # Check if it's a dictionary
            return section.get(key_name, default_value)
        return default_value  # Return default if section is None or not a dict

    flat_args["trust_remote_code"] = safe_get("model", "trust_remote_code", False)
    flat_args["hnm_use_faiss"] = safe_get("hnm", "use_faiss", False)
    flat_args["use_lora"] = safe_get("lora", "use_lora", False)
    flat_args["dataloader_pin_memory"] = safe_get(
        "training", "dataloader_pin_memory", True
    )
    flat_args["use_fp16"] = safe_get("training", "use_fp16", False)
    flat_args["use_bf16"] = safe_get("training", "use_bf16", False)
    flat_args["torch_compile"] = safe_get("training", "torch_compile", False)


def _ensure_all_parser_args_present(
    flat_args: Dict[str, Any], parser: argparse.ArgumentParser
):
    """Ensures all arguments defined in the parser exist in flat_args, adding defaults if missing."""
    for action in parser._actions:
        dest_name = action.dest
        if dest_name not in flat_args:
            # Only add if truly missing, respect None if explicitly set previously
            # (though _flatten_nested_config currently skips None)
            flat_args[dest_name] = action.default


# --- Basic Fixtures ---


@pytest.fixture(scope="function")
def tmp_path_factory(tmp_path_factory):
    """Re-export pytest's tmp_path_factory if needed elsewhere."""
    return tmp_path_factory


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory unique to each test function."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


# --- Configuration Fixtures (Factories & Instances) ---


@pytest.fixture
def minimal_config_dict_factory(temp_output_dir: Path):
    """Factory to create minimal config dicts, ensuring unique output dir."""

    def _factory(**overrides):
        # Base config with only STRICTLY required fields for AppSettings validation
        base = {
            "log_level": "DEBUG",
            "model": {"model_name_or_path": "mock-model-name"},
            "dataset": {
                "dataset_format": "pair",
                "dataset_name": "mock-hub-dataset",  # Add default source
            },
            "training": {"output_dir": str(temp_output_dir)},
        }
        # Simple merge, could be deeper if needed
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key].update(value)
            else:
                base[key] = value
        return base

    return _factory


@pytest.fixture
def minimal_config_dict(minimal_config_dict_factory) -> dict:
    """Provides a standard minimal config dict."""
    return minimal_config_dict_factory()


@pytest.fixture
def full_config_dict(minimal_config_dict: dict) -> dict:
    """Provides a more complete config dict with defaults filled in."""
    from finetune_embedding.config.settings import (
        AppSettings,
        DatasetConfig,
        ModelConfig,
        TrainingConfig,
    )

    # Create a default AppSettings instance to get all defaults
    default_settings = AppSettings(
        model=ModelConfig(model_name_or_path="dummy"),
        dataset=DatasetConfig(
            dataset_format="pair", dataset_name="dummy"
        ),  # Add dataset_name
        training=TrainingConfig(output_dir="dummy"),
    ).model_dump()  # Use model_dump()

    # Deep merge minimal config over defaults
    def deep_merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    merged = deep_merge(minimal_config_dict, default_settings)
    return merged


@pytest.fixture
def basic_app_settings(minimal_config_dict: dict):
    """Creates a basic AppSettings object from minimal config dict."""
    from finetune_embedding.config.settings import AppSettings

    return AppSettings(**minimal_config_dict)


@pytest.fixture
def full_app_settings(full_config_dict: dict):
    """Creates a more complete AppSettings object."""
    from finetune_embedding.config.settings import AppSettings

    return AppSettings(**full_config_dict)


@pytest.fixture
def mock_args_namespace(full_config_dict: dict) -> argparse.Namespace:
    """Creates a mock argparse.Namespace similar to parsed args by flattening the config."""
    flat_args: Dict[str, Any] = {}

    # 1. Add top-level args explicitly
    flat_args["config_file"] = None
    flat_args["log_level"] = full_config_dict.get("log_level", "INFO")
    flat_args["log_file"] = full_config_dict.get("log_file")

    # 2. Flatten nested sections (model, dataset, training, hnm, lora)
    for section, section_cfg in full_config_dict.items():
        if isinstance(section_cfg, dict):
            # Pass the section name to handle prefixes correctly inside the helper
            flat_args.update(_flatten_nested_config(section_cfg, parent_key=section))

    # 3. Add boolean flags explicitly
    _add_explicit_boolean_flags(flat_args, full_config_dict)

    # 4. Ensure all parser arguments exist, adding defaults if needed
    from finetune_embedding.config.loaders import _create_arg_parser

    parser = _create_arg_parser()
    _ensure_all_parser_args_present(flat_args, parser)

    # 5. Create the Namespace
    return argparse.Namespace(**flat_args)


# --- Mocking Fixtures (Factories & Instances) ---


@pytest.fixture
def mock_sentence_transformer_instance(mocker):
    """Provides a mocked instance of SentenceTransformer that evaluates to True."""
    mock_instance = MagicMock(spec=SentenceTransformer)
    mock_instance.encode.return_value = [[0.1, 0.2]]  # Example encode output
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_instance.device = torch.device("cpu")
    mock_instance.save = MagicMock()
    mock_instance.add_adapter = MagicMock()
    # Mock internal structure for PEFT checks
    mock_auto_model = MagicMock()
    mock_first_module = MagicMock()
    mock_first_module.auto_model = mock_auto_model
    mock_instance._first_module.return_value = mock_first_module

    # Configure the similarity method
    mock_instance.similarity.return_value = torch.tensor(
        [[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]]
    )  # Example 2x3 tensor

    # Configure __len__ to ensure truthiness
    mock_instance.__len__.return_value = 1

    return mock_instance


@pytest.fixture
def mock_load_dataset(mocker):
    """Mocks the datasets.load_dataset function."""
    return mocker.patch("datasets.load_dataset", autospec=True)


@pytest.fixture
def mock_sbert_trainer_instance(mocker):
    """Provides a mocked instance of SentenceTransformerTrainer."""
    mock_instance = MagicMock(spec=SentenceTransformerTrainer)
    mock_instance.train = MagicMock(
        return_value=MagicMock(training_loss=0.1)
    )  # Return mock result
    mock_instance.save_model = MagicMock()
    # Use spec for args as well
    mock_args = MagicMock(spec=SentenceTransformerTrainingArguments)
    mock_args.device = torch.device("cpu")
    mock_args.output_dir = "mock-output"
    mock_args.eval_strategy = "no"  # Default mock to no eval
    mock_args.load_best_model_at_end = False
    mock_instance.args = mock_args
    return mock_instance


@pytest.fixture
def mock_training_wrapper_instance(mocker, mock_sbert_trainer_instance):
    """Provides a mocked instance of TrainingWrapper."""
    # Mock the class to control instance creation
    mock_cls = mocker.patch(
        "finetune_embedding.training.trainer.TrainingWrapper", autospec=True
    )
    # Configure the instance returned by the class constructor
    mock_instance = mock_cls.return_value
    mock_instance.train = MagicMock()
    mock_instance.save_final_model = MagicMock(return_value="mock/output/final_model")
    # Link the internal trainer if needed for checks
    mock_instance.trainer = mock_sbert_trainer_instance
    mock_instance.training_args = mock_sbert_trainer_instance.args
    return mock_instance


# Removed mock_mine_hard_negatives fixture as it's mocked locally in test_hnm.py


@pytest.fixture
def mock_evaluator_instance(mocker):
    """Provides a mocked instance of a SentenceEvaluator."""
    # --- FIX: Remove spec ---
    # mock_instance = MagicMock(spec=evaluation.SentenceEvaluator)
    mock_instance = MagicMock()
    # --- END FIX ---
    # Simulate evaluator call returning a score or dict
    mock_instance.return_value = 0.85  # Example score
    mock_instance.name = "mock_evaluator"  # Keep name for logging/results
    return mock_instance


# --- Data Fixtures (Factories & Instances) ---
@pytest.fixture
def sample_dataset_dict() -> DatasetDict:
    """Provides a sample DatasetDict with various splits."""
    train_pair = Dataset.from_dict(
        {"sentence1": ["s1a", "s2a"], "sentence2": ["s1b", "s2b"]}
    )
    eval_pair_class = Dataset.from_dict(
        {"sentence1": ["e1a", "e2a"], "sentence2": ["e1b", "e2b"], "label": [1, 0]}
    )
    test_triplet = Dataset.from_dict(
        {"anchor": ["t1a"], "positive": ["t1p"], "negative": ["t1n"]}
    )
    return DatasetDict(
        {"train": train_pair, "validation": eval_pair_class, "test": test_triplet}
    )


@pytest.fixture
def sample_csv_file_factory(tmp_path_factory):
    """Factory to create sample CSV files."""

    def _factory(
        filename="sample_data.csv",
        content="col_A,col_B,label\ntext1,text2,0\ntext3,text4,1",
    ):
        # tmp_path_factory is session-scoped, mktemp creates a unique dir per call
        file_path = tmp_path_factory.mktemp("data") / filename
        file_path.write_text(content)
        return file_path

    return _factory


@pytest.fixture
def sample_json_file_factory(tmp_path_factory):
    """Factory to create sample JSON config files."""

    def _factory(filename="sample_config.json", content=None):
        if content is None:
            content = {"model": {"name": "test"}}  # Create default dict here
        file_path = tmp_path_factory.mktemp("config") / filename
        with open(file_path, "w") as f:
            json.dump(content, f)
        return file_path

    return _factory
