"""Utilities để save và load model checkpoints."""
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def save_checkpoint(
    model_state_dict,
    config,
    scaler,
    metrics,
    train_losses,
    val_losses,
    save_path
):
    """
    Save model checkpoint với tất cả thông tin cần thiết.
    
    Args:
        model_state_dict: Model state dict
        config: Configuration dict
        scaler: StandardScaler đã fit
        metrics: Dict chứa metrics (mape, etc.)
        train_losses: List training losses
        val_losses: List validation losses
        save_path: Đường dẫn để save (Path hoặc str)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config để có thể serialize (device không thể serialize)
    config_serializable = config.copy()
    if 'device' in config_serializable:
        config_serializable['device'] = str(config_serializable['device'])
    
    # Lưu scaler parameters thay vì object (để tương thích với weights_only=True)
    scaler_params = {
        'mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        'var_': scaler.var_.tolist() if hasattr(scaler, 'var_') else None,
        'n_features_in_': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None,
        'feature_names_in_': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None,
    }
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'config': config_serializable,
        'scaler_params': scaler_params,  # Lưu params thay vì object
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to: {save_path}")


def load_checkpoint(load_path, device=None):
    """
    Load model checkpoint.
    
    Args:
        load_path: Đường dẫn tới checkpoint file
        device: torch device (nếu None, tự động detect)
    
    Returns:
        Dict chứa: model_state_dict, config, scaler, metrics, train_losses, val_losses
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    
    # Load với weights_only=False để tương thích với các checkpoint cũ
    # Nếu muốn an toàn hơn, có thể dùng weights_only=True nhưng cần lưu scaler params
    try:
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch < 2.6 không có weights_only argument
        checkpoint = torch.load(load_path, map_location=device)
    
    # Restore device trong config
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'device' in checkpoint['config']:
        checkpoint['config']['device'] = device
    
    # Reconstruct scaler từ params
    if 'scaler_params' in checkpoint:
        scaler_params = checkpoint['scaler_params']
        
        # Tạo scaler mới và set attributes trực tiếp
        scaler = StandardScaler()
        
        if scaler_params['mean_'] is not None:
            mean_ = np.array(scaler_params['mean_'], dtype=np.float64)
            scaler.mean_ = mean_
        if scaler_params['scale_'] is not None:
            scale_ = np.array(scaler_params['scale_'], dtype=np.float64)
            scaler.scale_ = scale_
        if scaler_params['var_'] is not None:
            var_ = np.array(scaler_params['var_'], dtype=np.float64)
            scaler.var_ = var_
        if scaler_params['n_features_in_'] is not None:
            scaler.n_features_in_ = scaler_params['n_features_in_']
        if scaler_params['feature_names_in_'] is not None:
            scaler.feature_names_in_ = np.array(scaler_params['feature_names_in_'])
        
        # Đảm bảo scaler có thể transform được
        # Bằng cách set n_samples_seen_ (cần thiết cho một số version của sklearn)
        if hasattr(scaler, 'n_samples_seen_'):
            scaler.n_samples_seen_ = scaler.n_features_in_ if scaler.n_features_in_ else 1
        
        checkpoint['scaler'] = scaler
    elif 'scaler' in checkpoint:
        # Fallback: nếu checkpoint cũ vẫn có scaler object
        # Cần load với weights_only=False (đã làm ở trên)
        pass
    else:
        raise ValueError("Checkpoint không chứa scaler information")
    
    print(f"✅ Checkpoint loaded from: {load_path}")
    return checkpoint

