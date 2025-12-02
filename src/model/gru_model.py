"""GRU model cho time series prediction."""
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """GRU model cho stock price prediction."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_len):
        """
        Args:
            input_size: Số features đầu vào
            hidden_size: Kích thước hidden layer
            num_layers: Số layers GRU
            dropout: Dropout rate
            output_len: Độ dài output sequence
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        """
        Args:
            x: Input tensor shape (batch, seq_len, input_size)
        
        Returns:
            Output tensor shape (batch, output_len)
        """
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Lấy output của timestep cuối
        out = self.fc(out)
        return out

