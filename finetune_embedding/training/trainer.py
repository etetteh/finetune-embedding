# finetune_embedding/training/trainer.py
import logging
import os
from typing import Optional

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    # evaluation # Keep this if other parts of the module use it generally
)

# --- FIX: Import specific evaluator classes ---
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    SentenceEvaluator,  # Base class if needed
    TripletEvaluator,
)

# Use absolute imports
from finetune_embedding.config.settings import DEFAULT_EVAL_SPLIT, TrainingConfig
from finetune_embedding.exceptions import ConfigurationError, TrainingError

# --- FIX: Import IntervalStrategy if comparing against it ---
# from transformers.trainer_utils import IntervalStrategy # Or from sentence_transformers.training_args

logger = logging.getLogger(__name__)


class TrainingWrapper:
    """Wraps the SentenceTransformerTrainer for easier integration."""

    def __init__(
        self,
        model: SentenceTransformer,
        loss: object,  # Loss function instance
        train_dataset: Dataset,
        training_args_config: TrainingConfig,
        eval_dataset: Optional[Dataset] = None,
        # --- FIX: Use imported base class for type hint ---
        dev_evaluator: Optional[SentenceEvaluator] = None,
        dataset_name: str = "dataset",  # For evaluator naming
        eval_split_name: str = DEFAULT_EVAL_SPLIT,  # For evaluator naming
    ):
        self.model = model
        self.loss = loss
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dev_evaluator = dev_evaluator
        self.training_args_config = training_args_config
        self.dataset_name = dataset_name
        self.eval_split_name = eval_split_name
        self.trainer: Optional[SentenceTransformerTrainer] = None
        self.training_args: Optional[SentenceTransformerTrainingArguments] = None

    def _create_training_args(self) -> SentenceTransformerTrainingArguments:
        """Creates the SentenceTransformerTrainingArguments."""
        logger.info("Creating training arguments...")
        output_dir = self.training_args_config.output_dir
        if not output_dir:
            raise ConfigurationError("output_dir is required for training.")

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory '{output_dir}': {e}")
            raise ConfigurationError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e

        run_name = (
            self.training_args_config.run_name
            or os.path.basename(str(output_dir).rstrip(os.sep))
            or "sbert-finetune-run"
        )

        metric_for_best_model = self.training_args_config.metric_for_best_model
        greater_is_better = True  # Default assumption

        # Infer metric only if evaluating and user hasn't specified one
        if (
            self.training_args_config.eval_strategy != "no"
            and self.dev_evaluator
            and not metric_for_best_model
        ):
            eval_base_name = f"eval_{self.dataset_name}-{self.eval_split_name}"
            # --- FIX: Use imported classes for isinstance checks ---
            if isinstance(self.dev_evaluator, TripletEvaluator):
                metric_for_best_model = f"{eval_base_name}_cosine_accuracy"
            elif isinstance(self.dev_evaluator, EmbeddingSimilarityEvaluator):
                metric_for_best_model = (
                    f"{eval_base_name}_cosine_similarity"  # Spearman is common
                )
            elif isinstance(self.dev_evaluator, BinaryClassificationEvaluator):
                metric_for_best_model = f"{eval_base_name}_cosine_accuracy"
            else:
                logger.warning(
                    "Could not infer default metric for best model from evaluator type."
                )

            if metric_for_best_model:
                logger.info(
                    f"Defaulting metric_for_best_model to '{metric_for_best_model}' (greater_is_better={greater_is_better})"
                )

        # --- FIX: Handle eval_strategy override correctly ---
        effective_eval_strategy = self.training_args_config.eval_strategy
        if effective_eval_strategy != "no" and not self.dev_evaluator:
            logger.warning(
                f"Evaluation strategy is '{effective_eval_strategy}' but no dev evaluator available. Changing strategy to 'no'."
            )
            effective_eval_strategy = "no"  # Override strategy if no evaluator

        args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.training_args_config.epochs,
            per_device_train_batch_size=self.training_args_config.train_batch_size,
            per_device_eval_batch_size=self.training_args_config.eval_batch_size,
            learning_rate=self.training_args_config.learning_rate,
            warmup_ratio=self.training_args_config.warmup_ratio,
            lr_scheduler_type=self.training_args_config.lr_scheduler_type,
            weight_decay=self.training_args_config.weight_decay,
            gradient_accumulation_steps=self.training_args_config.gradient_accumulation_steps,
            max_grad_norm=self.training_args_config.max_grad_norm,
            seed=self.training_args_config.seed,
            fp16=self.training_args_config.use_fp16,
            bf16=self.training_args_config.use_bf16,
            # batch_sampler=BatchSamplers.NO_DUPLICATES, # Default in newer versions
            eval_strategy=effective_eval_strategy,  # Use the potentially overridden strategy
            eval_steps=self.training_args_config.eval_steps
            if effective_eval_strategy == "steps"
            else None,
            save_strategy=self.training_args_config.save_strategy,
            save_steps=self.training_args_config.save_steps
            if self.training_args_config.save_strategy == "steps"
            else None,
            save_total_limit=self.training_args_config.save_limit,
            logging_strategy=self.training_args_config.logging_strategy,
            logging_steps=self.training_args_config.logging_steps
            if self.training_args_config.logging_strategy == "steps"
            else None,
            dataloader_num_workers=self.training_args_config.dataloader_num_workers,
            dataloader_pin_memory=self.training_args_config.dataloader_pin_memory,
            torch_compile=self.training_args_config.torch_compile,
            report_to=self.training_args_config.report_to,
            run_name=run_name,
            load_best_model_at_end=bool(
                metric_for_best_model and self.dev_evaluator
            ),  # Only load best if metric exists AND evaluator exists
            metric_for_best_model=metric_for_best_model
            if self.dev_evaluator
            else None,  # Only set metric if evaluator exists
            greater_is_better=greater_is_better,
        )
        self.training_args = args
        logger.info("Training arguments created successfully.")
        logger.debug(f"TrainingArguments: {args.to_dict()}")
        return args

    def train(self) -> None:
        """Initializes the Trainer and runs the training loop."""
        logger.info("Initializing and starting trainer...")
        if not self.model:
            raise TrainingError("Model not initialized.")
        if not self.train_dataset:
            raise TrainingError("Train dataset not loaded.")
        if not self.loss:
            raise TrainingError("Loss function not created.")

        training_args = (
            self._create_training_args()
        )  # This now handles the eval_strategy override

        # No need to check/override eval_strategy again here, _create_training_args does it.

        self.trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if self.dev_evaluator else None,
            loss=self.loss,
            evaluator=self.dev_evaluator,
        )

        logger.info(f"Starting model training on device: {self.trainer.args.device}")
        try:
            train_result = self.trainer.train()
            logger.info("Training finished.")
            logger.debug(f"Train Result: {train_result}")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            if "OutOfMemoryError" in str(e) or "CUDA out of memory" in str(e):
                logger.error(
                    "Suggestion: Reduce batch sizes or increase gradient_accumulation_steps."
                )
            raise TrainingError("Training execution failed.") from e

    def save_final_model(self) -> str:
        """Saves the final model state after training."""
        if not self.trainer:
            raise TrainingError("Trainer not initialized. Cannot save model.")
        if not self.training_args:
            # Re-create args if they weren't stored (e.g., if train() wasn't called)
            # This might be needed if save_final_model is called independently
            self.training_args = self._create_training_args()
            # raise TrainingError("Training arguments not created. Cannot determine save path.")

        final_save_path = os.path.join(
            str(self.training_args.output_dir), "final_model"
        )
        logger.info(f"Saving final model state from trainer to {final_save_path}...")
        try:
            # Use trainer's save_model, which handles PEFT adapters correctly
            self.trainer.save_model(final_save_path)
            logger.info(f"Final model saved to {final_save_path}")
            return final_save_path
        except Exception as e:
            logger.error(f"Failed to save final model: {e}", exc_info=True)
            raise TrainingError(f"Failed to save final model: {e}") from e
