"""
DataLoader utilities for time series forecasting.
Complete data pipeline from raw data to PyTorch DataLoaders.

Dựa trên recommendations từ notebook exploration:
- Use 'close_log' (log-transformed) for stability
- Use sequence lengths: 7, 30, 120, 480 days
- Forecast horizon: 7 days
- Split: 70% train, 15% val, 15% test (temporal)
"""

import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .dataset import TimeSeriesDataset
from .preprocesser import (
    load_and_preprocess,
    normalize_data,
    temporal_split,
    create_sequences,
)


class DataPipeline:
    """
    Complete data pipeline for time series forecasting.
    
    Handles:
    - Data loading and preprocessing
    - Temporal train/val/test split
    - Normalization (only fit on training data!)
    - Sequence creation
    - DataLoader creation
    
    Dựa trên insights từ notebook exploration.
    """
    
    def __init__(
        self,
        data_path: str,
        target_col: str = "close_log",
        seq_lengths: List[int] = [7, 30, 120, 480],
        pred_len: int = 7,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
        use_log: bool = True,
        normalization_method: str = "standard",
    ):
        """
        Initialize data pipeline.
        
        Args:
            data_path: Path to CSV file
            target_col: Name of target column (after preprocessing, default: "close_log")
            seq_lengths: List of sequence lengths to create (default: [7, 30, 120, 480] from exploration)
            pred_len: Prediction length (default: 7 days from exploration)
            batch_size: Batch size for DataLoader
            train_ratio: Ratio of data for training (default: 70% from exploration)
            val_ratio: Ratio of data for validation (default: 15% from exploration)
            normalize: Whether to normalize data
            use_log: Whether to use log transformation (recommended from exploration)
            normalization_method: 'standard' or 'minmax'
        """
        self.data_path = data_path
        self.target_col = target_col
        self.seq_lengths = seq_lengths
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
        self.use_log = use_log
        self.normalization_method = normalization_method
        
        # Will be set after running pipeline
        self.scaler = None
        self.raw_data = None
        self.processed_data = None
        self.df = None
    
    def load_data(self) -> np.ndarray:
        """
        Load and preprocess data.
        
        Returns:
            Target data array
        """
        # Load and preprocess (dựa trên notebook exploration)
        self.df, target = load_and_preprocess(
            self.data_path,
            target_col="close",
            use_log=self.use_log,
        )
        
        self.raw_data = target
        
        return target
    
    def run(self) -> Dict[str, Dict[str, DataLoader]]:
        """
        Run complete pipeline.
        
        Returns:
            Dictionary of {seq_len: {split: dataloader}}
            Example: {'30d': {'train': train_loader, 'val': val_loader, 'test': test_loader}}
        """
        # Step 1: Load data
        target_data = self.load_data()
        
        # Step 2: Temporal split (QUAN TRỌNG: split theo thời gian!)
        train_data, val_data, test_data = temporal_split(
            target_data,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )
        
        # Step 3: Normalize (QUAN TRỌNG: chỉ fit trên training data!)
        if self.normalize:
            train_scaled, self.scaler = normalize_data(
                train_data,
                scaler=None,
                method=self.normalization_method,
            )
            val_scaled, _ = normalize_data(
                val_data,
                scaler=self.scaler,
                method=self.normalization_method,
            )
            test_scaled, _ = normalize_data(
                test_data,
                scaler=self.scaler,
                method=self.normalization_method,
            )
        else:
            train_scaled = train_data
            val_scaled = val_data
            test_scaled = test_data
        
        self.processed_data = {
            "train": train_scaled,
            "val": val_scaled,
            "test": test_scaled,
        }
        
        # Step 4: Create dataloaders for each sequence length
        dataloaders = {}
        
        for seq_len in self.seq_lengths:
            # Create sequences for each split
            X_train, y_train = create_sequences(train_scaled, seq_len, self.pred_len)
            X_val, y_val = create_sequences(val_scaled, seq_len, self.pred_len)
            X_test, y_test = create_sequences(test_scaled, seq_len, self.pred_len)
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            test_dataset = TimeSeriesDataset(X_test, y_test)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,  # Shuffle training data
                num_workers=0,
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Don't shuffle validation/test
                num_workers=0,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
            
            dataloaders[f"{seq_len}d"] = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }
        
        return dataloaders
    
    def get_scaler(self):
        """
        Get fitted scaler for denormalization.
        
        Returns:
            Fitted scaler object
        """
        return self.scaler
    
    def get_data_info(self) -> Dict:
        """
        Get information about processed data.
        
        Returns:
            Dictionary with data information
        """
        if self.processed_data is None:
            return {"status": "Pipeline not run yet"}
        
        info = {
            "raw_data_shape": len(self.raw_data) if self.raw_data is not None else None,
            "normalized": self.normalize,
            "scaler": self.scaler.__class__.__name__ if self.scaler else None,
            "use_log": self.use_log,
            "target_col": self.target_col,
            "splits": {},
        }
        
        for split_name, split_data in self.processed_data.items():
            info["splits"][split_name] = {
                "length": len(split_data),
                "mean": float(np.mean(split_data)),
                "std": float(np.std(split_data)),
                "min": float(np.min(split_data)),
                "max": float(np.max(split_data)),
            }
        
        return info
