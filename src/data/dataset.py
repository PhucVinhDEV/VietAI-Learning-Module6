"""
Dataset classes for time series forecasting.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting.
    
    Creates sequences of (input, target) pairs from time series data.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences of shape (n_samples, seq_len, 1)
            y: Target sequences of shape (n_samples, pred_len)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_sequence, target_sequence)
            - input_sequence: (seq_len, 1)
            - target_sequence: (pred_len,)
        """
        return self.X[idx], self.y[idx]
