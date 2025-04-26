# tests/training/test_losses.py
import pytest
from unittest.mock import MagicMock
from sentence_transformers import losses

from finetune_embedding.training.losses import create_loss_function
from finetune_embedding.exceptions import ConfigurationError, ModelError

# Use fixtures: mock_sentence_transformer_instance

@pytest.mark.parametrize("format, num_labels, expected_loss_type", [
    ("triplet", 2, losses.MultipleNegativesRankingLoss),
    ("pair-score", 2, losses.CoSENTLoss), # Or CosineSimilarityLoss, AnglELoss depending on default
    ("pair-class", 2, losses.SoftmaxLoss),
    ("pair-class", 3, losses.SoftmaxLoss),
])
def test_create_loss_function_selection(mock_sentence_transformer_instance, format, num_labels, expected_loss_type):
    """Test correct loss selection based on format."""
    loss = create_loss_function(mock_sentence_transformer_instance, format, num_labels)
    assert isinstance(loss, expected_loss_type)

def test_create_loss_function_no_model():
    """Test error if model is None."""
    with pytest.raises(ModelError, match="Model must be initialized"):
        create_loss_function(None, "triplet", 2)

def test_create_loss_function_invalid_format(mock_sentence_transformer_instance):
    """Test error for unsupported format."""
    with pytest.raises(ConfigurationError, match="Unsupported effective format"):
        create_loss_function(mock_sentence_transformer_instance, "invalid-format", 2)

def test_create_loss_function_softmax_invalid_labels(mock_sentence_transformer_instance):
    """Test error for SoftmaxLoss with insufficient labels."""
    with pytest.raises(ConfigurationError, match="SoftmaxLoss requires num_labels > 1"):
        create_loss_function(mock_sentence_transformer_instance, "pair-class", 1)

def test_create_loss_function_softmax_no_emb_dim(mock_sentence_transformer_instance):
    """Test error for SoftmaxLoss if embedding dim cannot be determined."""
    mock_sentence_transformer_instance.get_sentence_embedding_dimension.return_value = None
    with pytest.raises(ModelError, match="Could not determine embedding dimension"):
        create_loss_function(mock_sentence_transformer_instance, "pair-class", 2)

