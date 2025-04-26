# finetune_embedding/exceptions.py

class FineTuningError(Exception):
    """Base exception for the fine-tuning application."""
    pass

class ConfigurationError(FineTuningError):
    """Error related to configuration loading or validation."""
    pass

class DataLoadingError(FineTuningError):
    """Error related to dataset loading or processing."""
    pass

class ModelError(FineTuningError):
    """Error related to model initialization or handling."""
    pass

class TrainingError(FineTuningError):
    """Error occurring during the training process."""
    pass

class EvaluationError(FineTuningError):
    """Error occurring during evaluation."""
    pass
