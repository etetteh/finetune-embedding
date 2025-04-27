# finetune_embedding/config/loaders.py
import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

# Use absolute imports within the package
from finetune_embedding.config.settings import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_WARMUP_RATIO,
    AppSettings,
    DatasetConfig,
    HNMConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
)
from finetune_embedding.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def _load_config_from_file(filepath: str) -> Dict[str, Any]:
    """Loads configuration dictionary from a JSON file."""
    if not os.path.exists(filepath):
        logger.error(f"Configuration file not found: {filepath}")
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    try:
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return config_dict
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {filepath}: {e}")
        raise ConfigurationError(
            f"Invalid JSON in configuration file {filepath}"
        ) from e
    except OSError as e:
        logger.error(f"Could not read configuration file {filepath}: {e}")
        raise ConfigurationError(f"Could not read configuration file {filepath}") from e


def _create_arg_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with all configurable options."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentence transformer model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Config File Argument ---
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="JSON file containing configuration.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--log_file", type=str, default=None, help="Path to log file (optional)."
    )

    # --- Model Configuration ---
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path/name of the pre-trained model.",
    )
    model_group.add_argument(
        "--language", type=str, default="en", help="Model language for metadata."
    )
    model_group.add_argument(
        "--license_type", type=str, default="apache-2.0", help="Model license."
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading models with custom code.",
    )
    model_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching datasets and models.",
    )

    # --- Dataset Configuration ---
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name/path of the dataset (Hub ID or local path type like 'csv').",
    )
    data_group.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Specific configuration name for Hub datasets.",
    )
    data_group.add_argument(
        "--dataset_format",
        type=str,
        default=None,
        choices=["triplet", "pair", "pair-class", "pair-score"],
        help="Logical format of the *initial* dataset.",
    )
    data_group.add_argument(
        "--file_format",
        type=str,
        default=None,
        choices=["csv", "json", "text", "xml", "parquet"],
        help="Explicit format of local data files (e.g., 'csv', 'json').",
    )
    data_group.add_argument(
        "--data_files",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to local data files/glob patterns or dict mapping splits to files.",
    )
    data_group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing local data files.",
    )
    data_group.add_argument(
        "--column_rename_map",
        type=str,
        default=None,
        help="JSON string, path to JSON file, or JSON object mapping original to standard column names.",
    )
    data_group.add_argument(
        "--train_split",
        type=str,
        default=DEFAULT_TRAIN_SPLIT,
        help="Name of the training split.",
    )
    data_group.add_argument(
        "--eval_split",
        type=str,
        default=DEFAULT_EVAL_SPLIT,
        help="Name of the evaluation split.",
    )
    data_group.add_argument(
        "--test_split",
        type=str,
        default=DEFAULT_TEST_SPLIT,
        help="Name of the test split.",
    )
    data_group.add_argument(
        "--train_limit",
        type=int,
        default=0,
        help="Max examples for training (0 for all).",
    )
    data_group.add_argument(
        "--eval_limit",
        type=int,
        default=0,
        help="Max examples for evaluation (0 for all).",
    )
    data_group.add_argument(
        "--test_limit",
        type=int,
        default=0,
        help="Max examples for testing (0 for all).",
    )
    data_group.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of classes for 'pair-class' datasets (e.g., NLI has 3). Used for SoftmaxLoss.",
    )

    # --- Hard Negative Mining Configuration ---
    hnm_group = parser.add_argument_group(
        "Hard Negative Mining (HNM) Configuration (used if dataset_format='pair')"
    )
    hnm_group.add_argument(
        "--hnm_num_negatives",
        type=int,
        default=1,
        help="Number of hard negatives to mine per positive pair.",
    )
    hnm_group.add_argument(
        "--hnm_margin",
        type=float,
        default=None,
        help="Margin for HNM score difference.",
    )
    hnm_group.add_argument(
        "--hnm_range_min", type=int, default=0, help="Min rank of candidates for HNM."
    )
    hnm_group.add_argument(
        "--hnm_range_max",
        type=int,
        default=None,
        help="Max rank of candidates for HNM.",
    )
    hnm_group.add_argument(
        "--hnm_max_score",
        type=float,
        default=None,
        help="Max similarity score for HNM candidate.",
    )
    hnm_group.add_argument(
        "--hnm_min_score",
        type=float,
        default=None,
        help="Min similarity score for HNM candidate.",
    )
    hnm_group.add_argument(
        "--hnm_sampling_strategy",
        type=str,
        default="top",
        choices=["random", "top"],
        help="How to sample negatives.",
    )
    hnm_group.add_argument(
        "--hnm_use_faiss",
        action="store_true",
        help="Use FAISS for faster approximate search during HNM.",
    )

    # --- LoRA Configuration ---
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA parameter-efficient fine-tuning.",
    )
    lora_group.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    lora_group.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha scaling."
    )
    lora_group.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout probability."
    )
    lora_group.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="*",
        default=None,
        help="Specific modules for LoRA (e.g., 'query', 'key'). Default applies broadly.",
    )

    # --- Training Arguments ---
    train_group = parser.add_argument_group("Training Arguments")
    train_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and final model.",
    )
    train_group.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs."
    )
    train_group.add_argument(
        "--train_batch_size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Training batch size per device.",
    )
    train_group.add_argument(
        "--eval_batch_size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Evaluation batch size per device.",
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LR, help="Initial learning rate."
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Ratio of steps for linear warmup.",
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type.",
    )
    train_group.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps.",
    )
    train_group.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm."
    )
    train_group.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Evaluation frequency.",
    )
    train_group.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluate every N steps."
    )
    train_group.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Checkpoint saving frequency.",
    )
    train_group.add_argument(
        "--save_steps", type=int, default=100, help="Save checkpoint every N steps."
    )
    train_group.add_argument(
        "--save_limit", type=int, default=3, help="Max checkpoints to keep."
    )
    train_group.add_argument(
        "--logging_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Logging frequency.",
    )
    train_group.add_argument(
        "--logging_steps", type=int, default=100, help="Log every N steps."
    )
    train_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Num subprocesses for data loading.",
    )
    train_group.add_argument(
        "--no_dataloader_pin_memory",
        action="store_false",
        dest="dataloader_pin_memory",
        help="Disable pinning memory for data loading.",
    )
    train_group.set_defaults(dataloader_pin_memory=True)  # Default is True
    train_group.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 mixed precision (requires CUDA).",
    )
    train_group.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use BF16 mixed precision (requires Ampere+ GPU/MPS).",
    )
    train_group.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile optimization (PyTorch 2.0+).",
    )
    train_group.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed."
    )
    train_group.add_argument(
        "--report_to",
        type=str,
        nargs="*",
        default=None,
        help="Integrations for reporting (e.g., 'wandb', 'tensorboard'). 'none' to disable.",
    )
    train_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name for the run (e.g., for WandB).",
    )
    train_group.add_argument(
        "--metric_for_best_model",
        type=str,
        default=None,
        help="Metric name to determine the best model (e.g., 'eval_mydata-dev_cosine_accuracy').",
    )

    return parser


def _structure_config_dict(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a flat dictionary (from argparse) into a nested one for Pydantic."""
    # Initialize with top-level keys and empty dicts for nested models
    structured = {
        "log_file": flat_config.get("log_file"),
        "log_level": flat_config.get("log_level"),
        "model": {},
        "dataset": {},
        "hnm": {},
        "lora": {},
        "training": {},
    }

    # Helper to populate nested dicts, handling None values and prefixes
    def populate(target_dict: Dict, model_cls: type, prefix: str = ""):
        for field_name in model_cls.model_fields.keys():
            arg_name = f"{prefix}{field_name}"
            if arg_name in flat_config and flat_config[arg_name] is not None:
                target_dict[field_name] = flat_config[arg_name]

    populate(structured["model"], ModelConfig)
    populate(structured["dataset"], DatasetConfig)
    populate(structured["hnm"], HNMConfig, "hnm_")  # Add prefix for HNM args
    populate(structured["lora"], LoRAConfig, "lora_")  # Add prefix for LoRA args
    populate(structured["training"], TrainingConfig)

    # Handle boolean flags explicitly (argparse stores True/False)
    # These might not be present in flat_config if not specified and default is False
    structured["model"]["trust_remote_code"] = flat_config.get(
        "trust_remote_code", False
    )
    structured["hnm"]["use_faiss"] = flat_config.get("hnm_use_faiss", False)
    structured["lora"]["use_lora"] = flat_config.get("use_lora", False)
    structured["training"]["dataloader_pin_memory"] = flat_config.get(
        "dataloader_pin_memory", True
    )  # Default is True
    structured["training"]["use_fp16"] = flat_config.get("use_fp16", False)
    structured["training"]["use_bf16"] = flat_config.get("use_bf16", False)
    structured["training"]["torch_compile"] = flat_config.get("torch_compile", False)

    # Remove empty nested dicts if no relevant args were provided for them
    # This prevents Pydantic validation errors for missing required fields in nested models
    # if the entire section wasn't meant to be configured via CLI/defaults.
    # However, model, dataset, training are always required sections.
    if not structured.get("hnm"):
        structured.pop("hnm", None)
    if not structured.get("lora"):
        structured.pop("lora", None)

    return structured


# --- Refactored Helper Functions ---


def _parse_initial_args(
    parser: argparse.ArgumentParser,
) -> Tuple[Optional[str], List[str]]:
    """Parses known args to find config_file and return remaining args."""
    temp_args, remaining_argv = parser.parse_known_args()
    return temp_args.config_file, remaining_argv


def _load_config_file_if_present(
    config_file_path: Optional[str], parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    """Loads config from file if path is provided, handling errors."""
    if not config_file_path:
        return {}
    try:
        return _load_config_from_file(config_file_path)
    except (FileNotFoundError, ConfigurationError) as e:
        # Use parser.error to exit gracefully with help message
        parser.error(str(e))
        return {}  # Should not be reached due to parser.error


def _apply_config_file_to_defaults(
    config_from_file: Dict[str, Any], parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    """Overrides parser defaults with values from the config file."""
    defaults = {action.dest: action.default for action in parser._actions}
    for key, value in config_from_file.items():
        mapped_key = key  # Start with the key from the file
        # Add mappings if config file uses different names than args
        if key == "model_name" and "model_name_or_path" in defaults:
            mapped_key = "model_name_or_path"
        # Add more mappings as needed...

        if mapped_key in defaults:
            defaults[mapped_key] = value  # Update default value
            logger.debug(
                f"Config file setting default for '{mapped_key}' to '{value}'."
            )
        else:
            logger.warning(
                f"Key '{key}' from config file not found as a valid argument or mapping, ignoring."
            )
    return defaults


def _parse_final_args(
    parser: argparse.ArgumentParser,
    defaults: Dict[str, Any],
    remaining_argv: List[str],
) -> Dict[str, Any]:
    """Parses remaining CLI args, applying updated defaults."""
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    return vars(args)


def _validate_with_pydantic(structured_config: Dict[str, Any]) -> AppSettings:
    """Validates the structured config using the AppSettings Pydantic model."""
    try:
        # Ensure required top-level keys exist before passing to Pydantic
        # (Pydantic v2 might handle missing dicts better, but this is safer)
        structured_config.setdefault("model", {})
        structured_config.setdefault("dataset", {})
        structured_config.setdefault("training", {})

        app_settings = AppSettings(**structured_config)
        logger.info("Configuration parsed and validated successfully by Pydantic.")
        return app_settings
    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}", exc_info=False)
        # Provide a cleaner error message
        error_details = "\n".join(
            [f"  - {err['loc']}: {err['msg']}" for err in e.errors()]
        )
        raise ConfigurationError(
            f"Configuration validation failed:\n{error_details}"
        ) from e


def _perform_post_validation(app_settings: AppSettings):
    """Performs additional validation checks after Pydantic validation."""
    # Basic required field checks (redundant if Pydantic models are strict)
    if not app_settings.model.model_name_or_path:
        raise ConfigurationError("--model_name_or_path is required.")
    if not app_settings.training.output_dir:
        raise ConfigurationError("--output_dir is required.")
    if not app_settings.dataset.dataset_format:
        raise ConfigurationError("--dataset_format is required.")

    # Dataset source validation
    ds_cfg = app_settings.dataset
    has_hub_source = bool(
        ds_cfg.dataset_name
        and not ds_cfg.file_format
        and not ds_cfg.data_files
        and not ds_cfg.data_dir
    )
    has_local_source = bool(
        ds_cfg.file_format and (ds_cfg.data_files or ds_cfg.data_dir)
    )
    # Allows specifying dataset_name like 'csv' with local files
    has_local_hub_name_source = bool(
        ds_cfg.dataset_name
        and not ds_cfg.file_format
        and (ds_cfg.data_files or ds_cfg.data_dir)
    )

    if not (has_hub_source or has_local_source or has_local_hub_name_source):
        raise ConfigurationError(
            "Must provide dataset source: either --dataset_name (Hub) OR "
            "--file_format and (--data_files or --data_dir) (local) OR "
            "--dataset_name (as format type) and (--data_files or --data_dir)."
        )

    # Normalize 'report_to' if 'none' is included
    if app_settings.training.report_to and "none" in app_settings.training.report_to:
        app_settings.training.report_to = None  # Standard way to disable reporting

    logger.info("Post-Pydantic configuration validation passed.")


# --- Main Refactored Function ---


def load_and_validate_config() -> AppSettings:
    """
    Parses CLI arguments, loads optional config file and .env, validates,
    and returns the final AppSettings object. Orchestrates helper functions.
    """
    try:
        load_dotenv()
        logger.info(".env file loaded if found.")

        parser = _create_arg_parser()

        # 1. Parse initial args to find config file
        config_file_path, remaining_argv = _parse_initial_args(parser)

        # 2. Load config from file if specified
        config_from_file = _load_config_file_if_present(config_file_path, parser)

        # 3. Apply config file values to parser defaults
        updated_defaults = _apply_config_file_to_defaults(config_from_file, parser)

        # 4. Parse final args (CLI overrides config file/defaults)
        final_flat_config = _parse_final_args(parser, updated_defaults, remaining_argv)

        # 5. Structure the flat config for Pydantic
        structured_config = _structure_config_dict(final_flat_config)
        logger.debug(f"Structured config for Pydantic: {structured_config}")

        # 6. Validate with Pydantic
        app_settings = _validate_with_pydantic(structured_config)

        # 7. Perform post-Pydantic validation
        _perform_post_validation(app_settings)

        return app_settings

    except Exception as e:
        # Catch any unexpected errors during the process
        if not isinstance(e, ConfigurationError):
            logger.error(
                f"Unexpected error during configuration loading: {e}", exc_info=True
            )
            raise ConfigurationError(
                f"Unexpected error during configuration loading: {e}"
            ) from e
        else:
            # Re-raise known configuration errors
            raise e
