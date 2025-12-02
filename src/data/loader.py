"""Data loading utilities."""
from pathlib import Path
import pandas as pd
import numpy as np


def load_fpt_data(data_path=None):
    """
    Load FPT stock data từ CSV file.
    
    Args:
        data_path: Đường dẫn tới file CSV. Nếu None, tự động tìm trong project.
    
    Returns:
        DataFrame với columns: time, open, high, low, close, volume, symbol
    """
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {data_path}")
    else:
        # Tự động tìm file trong project
        cwd = Path.cwd()
        
        # Thử 2 vị trí chuẩn:
        # 1) Chạy từ root project: data/raw/FPT_train.csv
        # 2) Chạy từ thư mục notebooks: ../data/raw/FPT_train.csv
        candidates = [
            cwd / "data" / "raw" / "FPT_train.csv",
            cwd.parent / "data" / "raw" / "FPT_train.csv",
        ]
        
        path = None
        for p in candidates:
            if p.exists():
                path = p
                break
        
        # Fallback: rglob từ root project
        if path is None:
            project_root = cwd.parent if cwd.name == "notebooks" else cwd
            found = list(project_root.rglob("FPT_train.csv"))
            if not found:
                raise FileNotFoundError("Không tìm thấy FPT_train.csv trong project")
            path = found[0]
    
    print(f"Reading: {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    return df


def prepare_data(df):
    """
    Chuẩn bị data cho training: thêm log transform và filter columns.
    
    Args:
        df: DataFrame với time và close columns
    
    Returns:
        DataFrame với columns: time, close, close_log
    """
    df = df.copy()
    df["close_log"] = np.log(df["close"])
    df = df[["time", "close", "close_log"]].dropna().reset_index(drop=True)
    return df

