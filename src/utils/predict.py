"""Prediction utilities."""
import math
import numpy as np
import torch


def predict_future(model, history, scaler, input_len, total_days, device, clip_range=(0.8, 1.25)):
    """
    Predict giá cổ phiếu trong tương lai.
    
    Args:
        model: Trained PyTorch model
        history: List các giá trị close_log đã có
        scaler: StandardScaler đã fit
        input_len: Độ dài input sequence
        total_days: Số ngày cần predict
        device: torch device
        clip_range: Tuple (min_ratio, max_ratio) để clip giá trị
    
    Returns:
        List các giá trị close price đã predict
    """
    model.eval()
    pred_close = []
    history = history.copy()  # Copy để không modify original
    
    for day in range(total_days):
        # Lấy window cuối cùng
        window = np.array(history[-input_len:]).reshape(-1, 1)
        window_scaled = scaler.transform(window)
        
        # Predict
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_log = model(x).item()
        
        # Convert về giá thực
        last_close = math.exp(history[-1])
        new_close = math.exp(pred_log)
        
        # Clip để tránh giá trị quá lớn/nhỏ
        new_close = float(np.clip(new_close, clip_range[0] * last_close, clip_range[1] * last_close))
        
        pred_close.append(new_close)
        history.append(math.log(new_close))
    
    return pred_close


def evaluate_model(model, val_loader, device):
    """
    Evaluate model trên validation set.
    
    Args:
        model: Trained PyTorch model
        val_loader: DataLoader cho validation
        device: torch device
    
    Returns:
        mape: Mean Absolute Percentage Error
        preds: Predicted values
        true: True values
    """
    model.eval()
    preds_log = []
    true_log = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze().cpu().numpy()
            true = yb.squeeze().numpy()
            preds_log.append(pred)
            true_log.append(true)
    
    preds_log = np.concatenate(preds_log)
    true_log = np.concatenate(true_log)
    
    preds = np.exp(preds_log)
    true = np.exp(true_log)
    
    mape = np.mean(np.abs((true - preds) / true)) * 100
    
    return mape, preds, true

