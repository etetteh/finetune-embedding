# finetune_embedding/main.py
import logging
import sys

# Setup basic logging BEFORE loading config to catch config errors
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("finetune_embedding_cli")  # Specific logger for CLI phase

try:
    # Use absolute imports now
    from finetune_embedding.config.loaders import load_and_validate_config
    from finetune_embedding.exceptions import (
        ConfigurationError,
        DataLoadingError,
        FineTuningError,
    )
    from finetune_embedding.services.pipeline import FineTuningService
    from finetune_embedding.utils.logging import (
        setup_logging,
    )  # Import the setup function
except ImportError as e:
    logger.critical(
        f"Failed to import necessary modules. Ensure the package structure is correct and dependencies are installed. Error: {e}",
        exc_info=True,
    )
    print(
        f"Import Error: {e}. Check installation and package structure.", file=sys.stderr
    )
    sys.exit(1)


def main() -> int:
    """Main entry point for the fine-tuning CLI."""
    exit_code = 0
    settings = None
    try:
        logger.info("Loading and validating configuration...")
        settings = load_and_validate_config()

        # Setup logging based on validated config (replaces basicConfig)
        setup_logging(settings.log_level, settings.log_file)
        app_logger = logging.getLogger("finetune_embedding")  # Get app's root logger
        app_logger.info("Logging reconfigured based on settings.")

        app_logger.info("Initializing fine-tuning service...")
        service = FineTuningService(settings)
        service.run_pipeline()

    except (ConfigurationError, DataLoadingError, FineTuningError) as e:
        current_logger = logging.getLogger("finetune_embedding") if settings else logger
        current_logger.critical(f"Execution failed: {e}", exc_info=False)
        exit_code = 1
    except Exception as e:
        current_logger = logging.getLogger("finetune_embedding") if settings else logger
        current_logger.critical(
            f"An unexpected critical error occurred: {e}", exc_info=True
        )
        exit_code = 1
    finally:
        logging.shutdown()

    print(f"\nScript finished with exit code {exit_code}.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
