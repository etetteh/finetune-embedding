# finetune_embedding/evaluation/runner.py
import json
import logging
import os

from sentence_transformers import SentenceTransformer, evaluation

# Use absolute imports

logger = logging.getLogger(__name__)


def run_evaluation(
    evaluator: evaluation.SentenceEvaluator,
    model: SentenceTransformer,
    output_path: str,  # Directory to save results
) -> None:
    """Runs a given evaluator and saves the results."""
    if not evaluator:
        logger.warning("No evaluator provided, skipping evaluation run.")
        return
    if not model:
        logger.warning("No model provided, skipping evaluation run.")
        return

    eval_name = getattr(evaluator, "name", "unknown_evaluator")
    logger.info(f"Running evaluation for '{eval_name}'...")

    try:
        # Ensure output path exists for evaluator results
        os.makedirs(output_path, exist_ok=True)
        results = evaluator(model, output_path=output_path)

        results_dict = {}
        if isinstance(results, float):
            results_dict = {f"{eval_name}_primary_score": results}
            logger.info(f"Evaluation primary score ({eval_name}): {results:.4f}")
        elif isinstance(results, dict):
            results_dict = results
            logger.info(f"Evaluation results ({eval_name}):")
            for k, v in results.items():
                logger.info(f"  {k}: {v:.4f}")
        else:
            logger.warning(
                f"Evaluator '{eval_name}' returned unexpected type: {type(results)}. Value: {results}"
            )
            results_dict = {f"{eval_name}_result": str(results)}

        results_filename = f"{eval_name}_results.json"
        results_filepath = os.path.join(output_path, results_filename)
        try:
            with open(results_filepath, "w") as f:
                json.dump(results_dict, f, indent=4, sort_keys=True)
            logger.info(
                f"Evaluation results for '{eval_name}' saved to {results_filepath}"
            )
        except (TypeError, OSError) as e:
            logger.warning(
                f"Could not save evaluation results to JSON '{results_filepath}': {e}",
                exc_info=True,
            )

    except Exception as e:
        logger.error(f"Evaluation run failed for '{eval_name}': {e}", exc_info=True)
        # Optionally re-raise as EvaluationError
        # raise EvaluationError(f"Evaluation run failed for '{eval_name}': {e}") from e
