# finetune_embedding/services/pipeline.py
import logging
import os
import torch
from typing import Optional

# Use absolute imports
from finetune_embedding.config.settings import AppSettings
from finetune_embedding.data.loaders import load_raw_datasets, DatasetContainer
from finetune_embedding.data.preprocessing import (
    parse_column_rename_map,
    apply_column_renaming,
    auto_split_dataset,
    select_and_limit_splits
)
from finetune_embedding.data.hnm import apply_hard_negative_mining
from finetune_embedding.model.loaders import initialize_model
from finetune_embedding.model.lora import add_lora_adapter
from finetune_embedding.training.losses import create_loss_function
from finetune_embedding.training.trainer import TrainingWrapper
from finetune_embedding.evaluation.evaluators import create_evaluator
from finetune_embedding.evaluation.runner import run_evaluation
from finetune_embedding.utils.seeding import set_seed
from finetune_embedding.utils.device import determine_device, update_precision_flags
from finetune_embedding.exceptions import (
    FineTuningError, ConfigurationError, DataLoadingError, ModelError, TrainingError, EvaluationError
)

logger = logging.getLogger(__name__)

class FineTuningService:
    """Orchestrates the fine-tuning pipeline."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.device: Optional[torch.device] = None

    def _setup(self):
        """Initial setup: seeding, device, save config."""
        set_seed(self.settings.training.seed)
        self.device = determine_device()
        update_precision_flags(self.settings.training, self.device)
        self._save_effective_config()
        logger.info(f"Setup complete. Running on device: {self.device}")
        logger.info(f"Effective FP16: {self.settings.training.use_fp16}, Effective BF16: {self.settings.training.use_bf16}")

    def _save_effective_config(self):
        """Saves the final, potentially modified, configuration."""
        output_dir = self.settings.training.output_dir
        config_path = os.path.join(str(output_dir), "effective_training_config.json")
        logger.info(f"Saving final effective configuration to {config_path}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(config_path, "w") as f:
                f.write(self.settings.model_dump_json(indent=4)) # Use model_dump_json for Pydantic v2+
            logger.info(f"Effective configuration saved.")
        except Exception as e:
            logger.warning(f"Failed to save configuration to {config_path}: {e}", exc_info=False)

    def run_pipeline(self):
        """Executes the full fine-tuning process by calling component functions."""
        logger.info("===== Starting Fine-Tuning Pipeline Service =====")
        model = None
        try:
            self._setup()

            logger.info("--- Stage: Data Handling ---")
            raw_datasets = load_raw_datasets(self.settings.dataset, self.settings.model.cache_dir)
            rename_map = parse_column_rename_map(self.settings.dataset.column_rename_map)
            renamed_datasets = apply_column_renaming(raw_datasets, rename_map)
            split_datasets = auto_split_dataset(renamed_datasets, self.settings.dataset, self.settings.training.seed)
            datasets_container = select_and_limit_splits(split_datasets, self.settings.dataset)
            effective_format = self.settings.dataset.dataset_format

            logger.info("--- Stage: Model Initialization ---")
            model = initialize_model(self.settings.model, self.settings.training, self.device)

            if self.settings.dataset.dataset_format == "pair":
                logger.info("--- Stage: Hard Negative Mining ---")
                datasets_container, effective_format = apply_hard_negative_mining(
                    datasets=datasets_container,
                    model=model,
                    hnm_config=self.settings.hnm,
                    train_batch_size=self.settings.training.train_batch_size,
                    dataset_config=self.settings.dataset
                )
                logger.info(f"Hard Negative Mining applied. Effective format is now '{effective_format}'.")
            else:
                logger.info("Skipping Hard Negative Mining (initial format not 'pair').")

            if self.settings.lora.use_lora:
                logger.info("--- Stage: Applying LoRA ---")
                add_lora_adapter(model, self.settings.lora)
            else:
                 logger.info("Skipping LoRA.")

            logger.info("--- Stage: Loss Creation ---")
            loss = create_loss_function(
                model=model,
                effective_format=effective_format,
                num_labels=self.settings.dataset.num_labels
            )

            logger.info("--- Stage: Evaluator Setup ---")
            dev_evaluator = create_evaluator(
                eval_dataset=datasets_container.eval_dataset,
                effective_format=effective_format,
                dataset_config=self.settings.dataset,
                eval_batch_size=self.settings.training.eval_batch_size,
                name_prefix="eval"
            )

            logger.info("--- Stage: Training ---")
            trainer_wrapper = TrainingWrapper(
                model=model,
                loss=loss,
                train_dataset=datasets_container.train_dataset,
                eval_dataset=datasets_container.eval_dataset,
                dev_evaluator=dev_evaluator,
                training_args_config=self.settings.training,
                dataset_name=self.settings.dataset.dataset_name or "local",
                eval_split_name=self.settings.dataset.eval_split # Pass the configured name
            )
            trainer_wrapper.train()
            final_model_path = trainer_wrapper.save_final_model()
            logger.info(f"Training complete. Final model saved to: {final_model_path}")

            logger.info("--- Stage: Test Evaluation ---")
            test_evaluator = create_evaluator(
                eval_dataset=datasets_container.test_dataset,
                effective_format=effective_format,
                dataset_config=self.settings.dataset,
                eval_batch_size=self.settings.training.eval_batch_size,
                name_prefix="test"
            )
            if test_evaluator:
                run_evaluation(
                    evaluator=test_evaluator,
                    model=model, # Use model state after training
                    output_path=str(self.settings.training.output_dir)
                )
            else:
                 logger.warning("Skipping final test evaluation (no test data or evaluator).")

            logger.info("===== Fine-Tuning Pipeline Service Finished Successfully =====")

        except (ConfigurationError, DataLoadingError, ModelError, TrainingError, EvaluationError, FineTuningError) as e:
            logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred in the pipeline: {e}", exc_info=True)
            raise FineTuningError(f"Unexpected pipeline error: {e}") from e
