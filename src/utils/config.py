"""Configuration và constants cho FPT stock prediction."""
import random
import torch
import numpy as np

SEED = 42


def set_seed(seed=SEED):
    """
    Set random seed cho tất cả các thư viện để đảm bảo reproducibility.
    
    Args:
        seed: Random seed value (default: SEED = 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu có nhiều GPU
    
    # Đảm bảo deterministic behavior (có thể chậm hơn một chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable cho Python hash randomization
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


CONFIG = {
    "input_len": 30,
    "output_len": 1,
    "total_predict_days": 100,
    "batch_size": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "num_epochs": 80,
    "early_stop_patience": 15,
    "val_size": 120,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

