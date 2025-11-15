"""
Data preprocessing utilities for time series forecasting.
Dựa trên insights từ notebook exploration.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_and_preprocess(
    file_path: str,
    target_col: str = "close",
    use_log: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and preprocess stock data.
    
    Dựa trên notebook exploration:
    - Convert time sang datetime
    - Sort theo thời gian (QUAN TRỌNG cho time series!)
    - Drop symbol column
    - Tạo close_log (log transformation) nếu use_log=True
    
    Args:
        file_path: Path to CSV file
        target_col: Name of target column (before log transform)
        use_log: Whether to use log transformation (recommended from exploration)
        
    Returns:
        Tuple of (processed_dataframe, target_array)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert time sang datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        # Sort theo thời gian (QUAN TRỌNG cho time series!)
        df = df.sort_values("time").reset_index(drop=True)
    
    # Drop symbol column (không cần - chỉ có VIC)
    if "symbol" in df.columns:
        df = df.drop("symbol", axis=1)
    
    # Tạo target column với log transformation (recommended from exploration)
    if use_log:
        df["close_log"] = np.log(df[target_col])
        target_col = "close_log"
    
    # Extract target
    target = df[target_col].values
    
    return df, target


def normalize_data(
    data: np.ndarray,
    scaler: Optional[object] = None,
    method: str = "standard",
) -> Tuple[np.ndarray, object]:
    """
    Normalize time series data.
    
    Args:
        data: Input data (1D array)
        scaler: Pre-fitted scaler (for test data)
        method: Normalization method ('standard' or 'minmax')
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if scaler is None:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit on data
        data_2d = data.reshape(-1, 1)
        scaler.fit(data_2d)
    
    # Transform
    data_2d = data.reshape(-1, 1)
    normalized = scaler.transform(data_2d).flatten()
    
    return normalized, scaler


def denormalize_data(
    data: np.ndarray,
    scaler: object,
) -> np.ndarray:
    """
    Denormalize time series data.
    
    Args:
        data: Normalized data (1D or 2D array)
        scaler: Fitted scaler
        
    Returns:
        Denormalized data
    """
    # Handle both 1D and 2D inputs
    if data.ndim == 1:
        data_2d = data.reshape(-1, 1)
        denormalized = scaler.inverse_transform(data_2d).flatten()
    else:
        denormalized = scaler.inverse_transform(data)
    
    return denormalized


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences from time series.
    
    Tạo sliding windows cho time series forecasting.
    
    Args:
        data: Time series data (1D array)
        seq_len: Input sequence length (look-back window)
        pred_len: Prediction length (forecast horizon)
        
    Returns:
        Tuple of (X, y) arrays
        - X: (n_samples, seq_len, 1)
        - y: (n_samples, pred_len)
    """
    n_samples = len(data) - seq_len - pred_len + 1
    
    if n_samples <= 0:
        raise ValueError(
            f"Not enough data! Need at least {seq_len + pred_len} samples, "
            f"got {len(data)}"
        )
    
    X = np.zeros((n_samples, seq_len, 1))
    y = np.zeros((n_samples, pred_len))
    
    for i in range(n_samples):
        # Input: từ i đến i+seq_len
        X[i, :, 0] = data[i:i + seq_len]
        # Target: từ i+seq_len đến i+seq_len+pred_len
        y[i] = data[i + seq_len:i + seq_len + pred_len]
    
    return X, y


def temporal_split(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data temporally (important for time series!).
    
    Dựa trên recommendations từ exploration:
    - Split: 70% train, 15% val, 15% test (temporal)
    
    QUAN TRỌNG: Phải split theo thời gian, không shuffle ngẫu nhiên!
    
    Args:
        data: Time series data (1D array)
        train_ratio: Ratio of data for training (default: 70%)
        val_ratio: Ratio of data for validation (default: 15%)
        # test_ratio = 1 - train_ratio - val_ratio (15%)
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
