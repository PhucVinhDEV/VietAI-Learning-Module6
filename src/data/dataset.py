"""Time series dataset cho PyTorch."""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Dataset cho time series prediction với sliding window."""
    
    def __init__(self, data, feature_cols, input_len, output_len,
                 target_col="close_log", scaler=None, fit_scaler=False):
        """
        Args:
            data: DataFrame với feature columns
            feature_cols: List các feature columns
            input_len: Độ dài input sequence
            output_len: Độ dài output sequence
            target_col: Tên target column
            scaler: StandardScaler đã fit (dùng cho validation/test)
            fit_scaler: Nếu True, fit scaler mới (dùng cho train)
        """
        self.input_len = input_len
        self.output_len = output_len
        self.feature_cols = feature_cols
        self.target_col = target_col

        X = data[feature_cols].values.astype(np.float32)
        y = data[target_col].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            if scaler is None:
                raise ValueError("scaler phải được cung cấp khi fit_scaler=False")
            X = self.scaler.transform(X)

        X_seq, y_seq = [], []
        for i in range(len(data) - input_len - output_len + 1):
            X_seq.append(X[i:i+input_len])
            y_seq.append(y[i+input_len:i+input_len+output_len])

        self.X_seq = np.array(X_seq)
        self.y_seq = np.array(y_seq)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return torch.tensor(self.X_seq[idx]), torch.tensor(self.y_seq[idx])

