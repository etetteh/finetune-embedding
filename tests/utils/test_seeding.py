# tests/utils/test_seeding.py
import random
import numpy as np
import torch
from unittest.mock import patch

# Use absolute import
from finetune_embedding.utils.seeding import set_seed

@patch('torch.manual_seed')
@patch('numpy.random.seed')
@patch('random.seed')
@patch('torch.cuda.manual_seed_all')
def test_set_seed_calls(mock_cuda_seed, mock_py_seed, mock_np_seed, mock_torch_seed):
    """Test that set_seed calls underlying seeding functions."""
    test_seed = 123
    with patch('torch.cuda.is_available', return_value=True):
        set_seed(test_seed)

    mock_py_seed.assert_called_once_with(test_seed)
    mock_np_seed.assert_called_once_with(test_seed)
    mock_torch_seed.assert_called_once_with(test_seed)
    mock_cuda_seed.assert_called_once_with(test_seed)

@patch('torch.manual_seed')
@patch('numpy.random.seed')
@patch('random.seed')
@patch('torch.cuda.manual_seed_all')
def test_set_seed_no_cuda(mock_cuda_seed, mock_py_seed, mock_np_seed, mock_torch_seed):
    """Test that set_seed doesn't call CUDA seed if unavailable."""
    test_seed = 456
    with patch('torch.cuda.is_available', return_value=False):
        set_seed(test_seed)

    mock_py_seed.assert_called_once_with(test_seed)
    mock_np_seed.assert_called_once_with(test_seed)
    mock_torch_seed.assert_called_once_with(test_seed)
    mock_cuda_seed.assert_not_called()
