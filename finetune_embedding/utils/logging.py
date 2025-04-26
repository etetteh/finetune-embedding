# finetune_embedding/utils/logging.py
import logging
import sys
import os
from typing import Optional # Import Optional

def setup_logging(log_level_str: str, log_file: Optional[str] = None) -> None:
    """
    Sets up logging configuration based on user settings.
    """
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                 os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a'))
        except OSError as e:
            # Use basicConfig logger temporarily if main logger setup fails
            logging.error(f"Failed to create log directory for {log_file}: {e}. Logging to console only.", exc_info=False)

    # Remove existing handlers for the root logger to avoid duplicates
    root_logger = logging.getLogger()
    # Check if handlers exist before trying to remove/close
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                # Log potential error during handler removal/close
                logging.warning(f"Error removing/closing handler: {e}", exc_info=False)


    logging.basicConfig(level=numeric_level, format=log_format, datefmt=date_format, handlers=handlers, force=True) # Use force=True to override

    # Suppress noisy logs from underlying libraries if desired
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # Get the application's root logger after basicConfig
    app_logger = logging.getLogger("finetune_embedding") # Use package name
    app_logger.info(f"Logging configured at level {log_level_str.upper()}." + (f" Logging to file: {log_file}" if log_file else ""))
