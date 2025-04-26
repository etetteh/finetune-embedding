# Fine-Tune Embedding

[![Python CI - FineTune Embedding](https://github.com/etetteh/finetune-embedding/actions/workflows/python-ci.yml/badge.svg)](https://github.com/etetteh/finetune-embedding/actions/workflows/python-ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Architecture Overview](#3-architecture-overview)
4.  [Setup and Installation](#4-setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
5.  [Usage](#5-usage)
    *   [Basic Command Structure](#basic-command-structure)
    *   [Configuration File](#configuration-file)
    *   [Command-Line Arguments](#command-line-arguments)
    *   [Examples](#examples)
6.  [Configuration Details](#6-configuration-details)
    *   [Model Configuration](#model-configuration)
    *   [Dataset Configuration](#dataset-configuration)
    *   [Training Configuration](#training-configuration)
    *   [Hard Negative Mining (HNM) Configuration](#hard-negative-mining-hnm-configuration)
    *   [LoRA Configuration](#lora-configuration)
7.  [Testing](#7-testing)
8.  [Contributing](#8-contributing)
9.  [License](#9-license)
10. [Contact](#10-contact)
11. [Acknowledgements](#11-acknowledgements)

## 1. Introduction

This project provides a Python application for fine-tuning pre-trained sentence embedding models (primarily from the `sentence-transformers` library) on custom datasets for specific downstream tasks like semantic similarity, classification, or information retrieval.

The application is designed to be configurable and modular, allowing users to easily adapt models to their specific data and requirements using techniques like Hard Negative Mining and LoRA.

## 2. Features

*   **Easy Configuration:** Define fine-tuning jobs via command-line arguments or a JSON configuration file, validated using Pydantic.
*   **Flexible Data Loading:** Load datasets from Hugging Face Hub or local files (CSV, JSON, Parquet, etc.).
*   **Multiple Dataset Formats:** Supports common sentence embedding task formats: `triplet`, `pair`, `pair-score`, `pair-class`.
*   **Data Preprocessing:** Includes automatic train/validation/test splitting, column renaming, and data limiting options.
*   **Hard Negative Mining (HNM):** Automatically mines hard negatives for `pair` datasets to improve contrastive learning.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Supports LoRA (Low-Rank Adaptation) for efficient fine-tuning.
*   **Appropriate Loss Functions:** Automatically selects suitable loss functions based on the effective dataset format.
*   **Flexible Evaluation:** Configurable evaluation during training and final evaluation on a test set using relevant metrics.
*   **Device & Precision Control:** Auto-detects and utilizes CUDA/MPS/CPU; handles FP16/BF16 precision.
*   **Reproducibility:** Allows setting random seeds for consistent results.
*   **Comprehensive Logging:** Configurable logging level and optional file output.

## 3. Architecture Overview

The project follows a modular structure to separate concerns:

finetune-embedding/ ├── tests/ # Unit and Integration Tests ├── finetune_embedding/ │ ├── init.py │ ├── main.py # Entry point, argument parsing, orchestration │ ├── config/ # Configuration models (Pydantic) and loading logic │ ├── data/ # Data loading, preprocessing, HNM logic │ ├── model/ # Model loading, LoRA adapter logic │ ├── training/ # Loss creation, Trainer wrapper │ ├── evaluation/ # Evaluator creation, evaluation running logic │ ├── services/ # Pipeline orchestration service │ └── utils/ # General utilities (logging, seeding, device) ├── .github/workflows/ # CI workflows ├── .env.example # Example environment file (if needed) ├── .gitignore ├── README.md └── pyproject.toml # Project metadata and dependencies


*   **`main.py`:** Handles the command-line interface, loads configuration, sets up logging, and initiates the main pipeline service.
*   **`config/`:** Defines Pydantic models for configuration validation and provides loaders for CLI arguments and JSON files.
*   **`data/`:** Manages dataset loading from various sources, preprocessing steps (splitting, renaming, limiting), and Hard Negative Mining.
*   **`model/`:** Handles the initialization of `SentenceTransformer` models and the application of LoRA adapters via the `peft` library.
*   **`training/`:** Selects appropriate loss functions and wraps the `SentenceTransformerTrainer` for the training loop.
*   **`evaluation/`:** Creates specific evaluators based on the task format and runs the evaluation process.
*   **`services/`:** Contains the `FineTuningService` which orchestrates the entire workflow from setup to final evaluation.
*   **`utils/`:** Provides shared utilities for logging setup, random seeding, and device/precision management.

## 4. Setup and Installation

### Prerequisites

*   Python 3.9+
*   `pip` and `venv` (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/etetteh/finetune-embedding.git
    cd finetune-embedding
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project uses `pyproject.toml` for dependency management. Install the package in editable mode along with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
    *Note: If you encounter issues with specific dependencies (like `torch` with CUDA), you might need to install them manually following instructions from their official websites (e.g., PyTorch).*

## 5. Usage

The script is run as a module from the project root directory.

### Basic Command Structure

```bash
python -m finetune_embedding [OPTIONS]
```


Use python -m finetune_embedding --help to see a full list of arguments and their defaults.

Configuration File
You can define all parameters in a JSON configuration file and pass it using --config_file.

### Example config.json:
```json
{
    "model": {
        "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "dataset": {
        "dataset_name": "stsb_multi_mt",
        "dataset_config_name": "en",
        "dataset_format": "pair-score"
    },
    "training": {
        "output_dir": "./output/stsb-finetuned-config",
        "epochs": 2,
        "train_batch_size": 16,
        "learning_rate": 2e-5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": true,
        "use_fp16": true
    }
}
```

### Run using config file:
```bash
python -m finetune_embedding --config_file config.json
```

## Command-Line Arguments
Alternatively, specify parameters directly via the command line. CLI arguments override values from the configuration file.

### Example using CLI arguments:

```bash
python -m finetune_embedding \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --dataset_name "stsb_multi_mt" \
    --dataset_config_name "en" \
    --dataset_format "pair-score" \
    --output_dir "./output/stsb-finetuned-cli" \
    --epochs 2 \
    --train_batch_size 16 \
    --learning_rate 2e-5 \
    --use_fp16
```

## Examples
### Fine-tune for Semantic Textual Similarity (STS):
```bash
python -m finetune_embedding \
    --model_name_or_path "sentence-transformers/all-MiniLM-L6-v2" \
    --dataset_name "stsb_multi_mt" \
    --dataset_format "pair-score" \
    --output_dir "./output/sts-finetuned" \
    --epochs 1 \
    --train_batch_size 16 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end true \
    --metric_for_best_model "eval_stsb_multi_mt-validation_cosine_similarity"
```

### Fine-tune using Triplets with LoRA:
```bash
# Assuming a dataset 'my-triplet-dataset' exists on the Hub or locally
python -m finetune_embedding \
    --model_name_or_path "sentence-transformers/msmarco-distilbert-base-v4" \
    --dataset_name "my-triplet-dataset" \
    --dataset_format "triplet" \
    --output_dir "./output/triplet-lora-finetuned" \
    --epochs 3 \
    --train_batch_size 32 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --learning_rate 5e-5 \
    --use_fp16
```

### Fine-tune using Pairs with Hard Negative Mining:

```bash
# Assuming a dataset 'my-pair-dataset' with columns 'sent1', 'sent2'
python -m finetune_embedding \
    --model_name_or_path "sentence-transformers/msmarco-distilbert-base-v4" \
    --dataset_name "my-pair-dataset" \
    --dataset_format "pair" \
    --column_rename_map '{"sent1": "sentence1", "sent2": "sentence2"}' \
    --output_dir "./output/pair-hnm-finetuned" \
    --epochs 2 \
    --train_batch_size 32 \
    --hnm_num_negatives 2 \
    --learning_rate 3e-5 \
    --use_fp16
```

## 6. Configuration Details
The fine-tuning process is controlled by several configuration sections, available both in the JSON config file and as CLI arguments (often prefixed).

### Model Configuration
(model in JSON, --model_name_or_path, etc. in CLI)

* model_name_or_path: Base sentence transformer model ID or path. (Required)
* language, license_type: Metadata for the model card.
* trust_remote_code: Allow custom code execution from the model hub.
* cache_dir: Optional directory for caching models/datasets.

### Dataset Configuration
(dataset in JSON, --dataset_name, etc. in CLI)

* dataset_name: Hub ID or local format type (e.g., 'csv').
* dataset_config_name: Specific configuration for Hub datasets.
* dataset_format: Logical format (triplet, pair, pair-score, pair-class). (Required)
* file_format, data_files, data_dir: For loading local datasets.
* column_rename_map: Map original column names to standard names (sentence1, sentence2, positive, negative, anchor, score, label). Can be a JSON string, file path, or dict.
* train_split, eval_split, test_split: Names of the splits to use.
* train_limit, eval_limit, test_limit: Limit the number of examples per split.
* num_labels: Number of classes for pair-class format (used by SoftmaxLoss).

### Training Configuration
(training in JSON, --output_dir, etc. in CLI)

* output_dir: Directory to save checkpoints and the final model. (Required)
* epochs, train_batch_size, eval_batch_size, learning_rate, warmup_ratio, lr_scheduler_type, weight_decay, gradient_accumulation_steps, max_grad_norm: Standard training hyperparameters.
* eval_strategy, eval_steps, save_strategy, save_steps, save_limit, logging_strategy, logging_steps: Control evaluation, saving, and logging frequency.
* dataloader_num_workers, dataloader_pin_memory: Dataloader performance settings.
* use_fp16, use_bf16: Enable mixed-precision training.
* torch_compile: Enable PyTorch 2.0+ compilation (experimental).
* seed: Random seed for reproducibility.
* report_to: Integrations like 'wandb', 'tensorboard'.
* run_name: Custom name for reporting runs.
* metric_for_best_model: Metric used to identify the best checkpoint during training.

### Hard Negative Mining (HNM) Configuration
(hnm in JSON, --hnm_* in CLI)

* Used only if dataset_format is pair.
* Controls parameters like num_negatives, margin, score ranges (min_score, max_score), sampling strategy, and FAISS usage (use_faiss).

### LoRA Configuration
(lora in JSON, --use_lora, --lora_* in CLI)

* use_lora: Enable LoRA fine-tuning.
* rank, alpha, dropout, target_modules: LoRA-specific hyperparameters.

## 7. Testing
The project includes a test suite using pytest.

1. Ensure development dependencies are installed: pip install -e ".[dev]"
2. Run linters and formatters:

```bash
flake8 .
black --check .
isort --check-only .
```

3. Run type checks:

```bash
mypy finetune_embedding/
```

4. Run tests:
```bash
pytest -v
```

5. Run tests with coverage:
```bash
pytest -v --cov=finetune_embedding --cov-report=term-missing
```

## 8. Contributing
Contributions are welcome! Please follow these general steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix (git checkout -b feature/your-feature-name).
3. Make your changes, adhering to the existing code style.
4. Add tests for your changes in the tests/ directory.
5. Ensure all tests pass (pytest -v).
6. Run linters and type checks to ensure code quality.
7. Commit your changes with clear messages (git commit -m 'Add your feature').
8. Push your branch to your fork (git push origin feature/your-feature-name).
9. Open a Pull Request to the main repository (etetteh/finetune-embedding).

## 9. License

This project is licensed under the Apache License 2.0. See the LICENSE file for details. 

## 10. Contact

For questions or issues, please open an issue on the GitHub repository: github.com/etetteh/finetune-embedding/issues

Project Author: Enoch Tetteh (@etetteh)

## 11. Acknowledgements

This project relies on several fantastic open-source libraries, including:

* Sentence Transformers
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face PEFT
* PyTorch
* Pydantic

/br
© 2024 Enoch Tetteh