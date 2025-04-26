# tests/evaluation/test_runner.py
import pytest
import json
import os
import logging 
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Use absolute imports
from finetune_embedding.evaluation.runner import run_evaluation
from finetune_embedding.exceptions import EvaluationError

# Fixtures from conftest.py:
# mock_evaluator_instance, mock_sentence_transformer_instance, temp_output_dir

def test_run_evaluation_success_float(mock_evaluator_instance, mock_sentence_transformer_instance, temp_output_dir):
    """Test running evaluation when evaluator returns a float."""
    mock_evaluator_instance.return_value = 0.95 # Simulate float return
    mock_evaluator_instance.name = "eval_test-split"

    with patch("builtins.open", mock_open()) as mock_file:
        run_evaluation(mock_evaluator_instance, mock_sentence_transformer_instance, str(temp_output_dir))

    mock_evaluator_instance.assert_called_once_with(mock_sentence_transformer_instance, output_path=str(temp_output_dir))
    # Check if results file was opened correctly
    mock_file.assert_called_once_with(os.path.join(temp_output_dir, "eval_test-split_results.json"), "w")
    handle = mock_file()

    # --- FIX: Concatenate all write calls ---
    all_written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    written_data = json.loads(all_written_content)
    # --- END FIX ---

    assert written_data == {"eval_test-split_primary_score": 0.95}

def test_run_evaluation_success_dict(mock_evaluator_instance, mock_sentence_transformer_instance, temp_output_dir):
    """Test running evaluation when evaluator returns a dict."""
    result_dict = {"accuracy": 0.8, "f1": 0.75}
    mock_evaluator_instance.return_value = result_dict
    mock_evaluator_instance.name = "eval_test-split"

    with patch("builtins.open", mock_open()) as mock_file:
        run_evaluation(mock_evaluator_instance, mock_sentence_transformer_instance, str(temp_output_dir))

    mock_evaluator_instance.assert_called_once_with(mock_sentence_transformer_instance, output_path=str(temp_output_dir))
    mock_file.assert_called_once_with(os.path.join(temp_output_dir, "eval_test-split_results.json"), "w")
    handle = mock_file()

    # --- FIX: Concatenate all write calls ---
    all_written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    written_data = json.loads(all_written_content)
    # --- END FIX ---

    assert written_data == result_dict

def test_run_evaluation_no_evaluator(mock_sentence_transformer_instance, temp_output_dir):
    """Test that nothing happens if evaluator is None."""
    with patch("builtins.open", mock_open()) as mock_file:
         # We don't expect the evaluator or open to be called
         run_evaluation(None, mock_sentence_transformer_instance, str(temp_output_dir))
    mock_file.assert_not_called()

def test_run_evaluation_no_model(mock_evaluator_instance, temp_output_dir):
    """Test that nothing happens if model is None."""
    with patch("builtins.open", mock_open()) as mock_file:
         run_evaluation(mock_evaluator_instance, None, str(temp_output_dir))
    mock_evaluator_instance.assert_not_called()
    mock_file.assert_not_called()

def test_run_evaluation_save_fails(mock_evaluator_instance, mock_sentence_transformer_instance, temp_output_dir, caplog):
    """Test that a warning is logged if saving results fails."""
    mock_evaluator_instance.return_value = 0.5
    mock_evaluator_instance.name = "eval_fail"
    # Simulate OSError during open
    with patch("builtins.open", side_effect=OSError("Disk full")) as mock_file:
        run_evaluation(mock_evaluator_instance, mock_sentence_transformer_instance, str(temp_output_dir))

    mock_evaluator_instance.assert_called_once() # Evaluator still runs
    mock_file.assert_called_once() # Attempt to open still happens
    assert "Could not save evaluation results" in caplog.text
    assert "Disk full" in caplog.text

def test_run_evaluation_eval_fails(mock_evaluator_instance, mock_sentence_transformer_instance, temp_output_dir, caplog):
    """Test that an error is logged if the evaluator call fails."""
    mock_evaluator_instance.side_effect = ValueError("Internal evaluator error")
    mock_evaluator_instance.name = "eval_error"

    caplog.set_level(logging.ERROR)

    with patch("builtins.open", mock_open()) as mock_file:
        # Expect the function to catch the error and log it, not crash
        run_evaluation(mock_evaluator_instance, mock_sentence_transformer_instance, str(temp_output_dir))

    mock_evaluator_instance.assert_called_once() # Attempt to call evaluator happens
    mock_file.assert_not_called() # File writing shouldn't happen if eval fails
    assert "Evaluation run failed for 'eval_error'" in caplog.text # Match actual log
    # --- END FIX ---
    assert "Internal evaluator error" in caplog.text # Also check the specific error is logged
