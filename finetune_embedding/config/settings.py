# finetune_embedding/config/settings.py
import logging
from pathlib import Path  # Import torch to check for compile availability
from typing import Dict, List, Literal, Optional, Union

import torch
from pydantic import (  # Group imports
    BaseModel,
    ConfigDict,
    Field,
    Json,
    field_validator,
)

# Define constants for defaults
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_TRAIN_PROP = 0.8
DEFAULT_EVAL_PROP = 0.1
DEFAULT_TEST_PROP = 0.1
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1

DatasetFormat = Literal["triplet", "pair", "pair-class", "pair-score"]
Strategy = Literal["steps", "epoch", "no"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ModelConfig(BaseModel):
    model_name_or_path: str = Field(
        ..., description="Path/name of the pre-trained model."
    )
    language: str = Field("en", description="Model language for metadata.")
    license_type: str = Field("apache-2.0", description="Model license.")
    trust_remote_code: bool = Field(
        False, description="Allow loading models with custom code."
    )
    cache_dir: Optional[Path] = Field(
        None, description="Directory for caching datasets and models."
    )


class DatasetConfig(BaseModel):
    dataset_name: Optional[str] = Field(
        None,
        description="Name/path of the dataset (Hub ID or local path type like 'csv').",
    )
    dataset_config_name: Optional[str] = Field(
        None, description="Specific configuration name for Hub datasets."
    )
    dataset_format: DatasetFormat = Field(
        ..., description="Logical format of the *initial* dataset."
    )
    file_format: Optional[Literal["csv", "json", "text", "xml", "parquet"]] = Field(
        None, description="Explicit format of local data files."
    )
    data_files: Optional[
        Union[
            List[Union[str, Path]], Dict[str, Union[str, Path, List[Union[str, Path]]]]
        ]
    ] = Field(
        None,
        description="Path(s) to local data files/glob patterns or dict mapping splits to files.",
    )
    data_dir: Optional[Path] = Field(
        None, description="Path to directory containing local data files."
    )
    column_rename_map: Optional[Union[Json, Path, Dict[str, str]]] = Field(
        None,
        description="JSON string, path to JSON file, or dict mapping original to standard column names.",
    )
    train_split: str = Field(
        DEFAULT_TRAIN_SPLIT, description="Name of the training split."
    )
    eval_split: str = Field(
        DEFAULT_EVAL_SPLIT, description="Name of the evaluation split."
    )
    test_split: str = Field(DEFAULT_TEST_SPLIT, description="Name of the test split.")
    train_limit: int = Field(
        0, ge=0, description="Max examples for training (0 for all)."
    )
    eval_limit: int = Field(
        0, ge=0, description="Max examples for evaluation (0 for all)."
    )
    test_limit: int = Field(
        0, ge=0, description="Max examples for testing (0 for all)."
    )
    num_labels: int = Field(
        2,
        gt=0,
        description="Number of classes for 'pair-class' datasets. Used for SoftmaxLoss.",
    )

    @field_validator("data_files")
    @classmethod
    def data_files_must_be_list_or_dict(cls, v):
        # Logic remains the same for this simple validator
        if v is not None and not isinstance(v, (list, dict)):
            raise ValueError("data_files must be a list or a dictionary")
        return v


class HNMConfig(BaseModel):
    num_negatives: int = Field(1, gt=0)
    margin: Optional[float] = None
    range_min: int = Field(0, ge=0)
    range_max: Optional[int] = Field(None, gt=0)
    max_score: Optional[float] = None
    min_score: Optional[float] = None
    sampling_strategy: Literal["random", "top"] = "top"
    use_faiss: bool = False


class LoRAConfig(BaseModel):
    use_lora: bool = False
    rank: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    target_modules: Optional[List[str]] = None


class TrainingConfig(BaseModel):
    # Use Path for output_dir, no DirectoryPath needed
    output_dir: Path = Field(
        ..., description="Directory to save checkpoints and final model."
    )
    epochs: int = Field(DEFAULT_EPOCHS, gt=0)
    train_batch_size: int = Field(DEFAULT_TRAIN_BATCH_SIZE, gt=0)
    eval_batch_size: int = Field(DEFAULT_EVAL_BATCH_SIZE, gt=0)
    learning_rate: float = Field(DEFAULT_LR, gt=0)
    warmup_ratio: float = Field(DEFAULT_WARMUP_RATIO, ge=0.0, le=1.0)
    lr_scheduler_type: str = "linear"
    weight_decay: float = Field(0.01, ge=0.0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    max_grad_norm: float = Field(1.0, gt=0)
    eval_strategy: Strategy = "steps"
    eval_steps: int = Field(100, gt=0)
    save_strategy: Strategy = "steps"
    save_steps: int = Field(100, gt=0)
    save_limit: int = Field(3, ge=1)
    logging_strategy: Strategy = "steps"
    logging_steps: int = Field(100, gt=0)
    dataloader_num_workers: int = Field(0, ge=0)
    dataloader_pin_memory: bool = True
    use_fp16: bool = False
    use_bf16: bool = False
    torch_compile: bool = False
    seed: int = DEFAULT_SEED
    report_to: Optional[List[str]] = None  # e.g., ["wandb", "tensorboard"]
    run_name: Optional[str] = None
    metric_for_best_model: Optional[str] = None

    # --- V2 Style Validator ---
    # No validator needed for output_dir existence check anymore

    # --- V2 Style Validator ---
    @field_validator("torch_compile")
    @classmethod
    def check_torch_compile_availability(cls, v):
        # Logic remains the same
        if v and not hasattr(torch, "compile"):
            logging.warning(
                "torch.compile requested but not available (requires PyTorch 2.0+). Disabling."
            )
            return False
        return v


class AppSettings(BaseModel):
    """Main configuration model combining all sections."""

    # Allow Path for log_file
    log_file: Optional[Path] = None
    log_level: LogLevel = "INFO"

    model: ModelConfig
    dataset: DatasetConfig
    hnm: HNMConfig = Field(default_factory=HNMConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig

    # --- V2 Style Config ---
    # If you need specific Pydantic config settings, use ConfigDict
    # For example, to allow extra fields (though usually not recommended):
    # model_config = ConfigDict(extra='ignore')
    # If you had `arbitrary_types_allowed = True`, it's often enabled by default in V2
    # or you can set it explicitly:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- V2 Style Cross-Field Validation ---
    # Use model_validator for checks involving multiple fields
    # This replaces the old @validator('hnm', always=True) pattern
    # Note: This requires importing `model_validator` from pydantic
    # from pydantic import model_validator

    # @model_validator(mode='after') # 'after' runs after individual field validation
    # def check_hnm_relevance_v2(self) -> 'AppSettings':
    #     if self.dataset.dataset_format != 'pair':
    #         default_hnm = HNMConfig()
    #         hnm_args_set = any(
    #             getattr(self.hnm, field) != getattr(default_hnm, field)
    #             for field in HNMConfig.model_fields # Use model_fields in V2
    #         )
    #         if hnm_args_set:
    #             logging.warning(f"Hard Negative Mining settings provided, but dataset_format='{self.dataset.dataset_format}' (not 'pair'). HNM settings will be ignored.")
    #     return self

    # @model_validator(mode='after')
    # def check_precision_conflict_v2(self) -> 'AppSettings':
    #     if self.training.use_fp16 and self.training.use_bf16:
    #         logging.warning("Both use_fp16 and use_bf16 requested. Device capabilities will determine final usage.")
    #     return self

    # --- OR keep simple field validators if they don't need other fields ---
    # This validator only needs the 'training' field itself
    @field_validator("training")
    @classmethod
    def check_precision_conflict_v2_simple(cls, training_config):
        if training_config.use_fp16 and training_config.use_bf16:
            logging.warning(
                "Both use_fp16 and use_bf16 requested. Device capabilities will determine final usage."
            )
        return training_config

    # This validator only needs the 'hnm' field itself (the warning logic is fine here)
    # However, the check against dataset_format *does* require another field,
    # so a model_validator is technically more correct for that part.
    # Let's keep the simple warning logic here for now.
    @field_validator("hnm")
    @classmethod
    def check_hnm_relevance_v2_simple(cls, hnm_config):
        # This validator runs *before* the dataset field might be validated.
        # The warning logic inside the validator might be less reliable here.
        # The logic inside the pipeline service that checks format before calling HNM
        # is a more robust place for the "ignore HNM" decision.
        # We can leave this validator out or just keep it simple.
        pass  # Or add simple validation on hnm_config fields themselves
        return hnm_config
