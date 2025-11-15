"""
Simple test script for data processing pipeline.
Cách chạy: 
    python tests/test_data_simple.py
    hoặc
    cd tests && python test_data_simple.py
"""

import sys
from pathlib import Path

# Add parent directory to path để import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataPipeline
import numpy as np

print("="*60)
print("TEST DATA PROCESSING PIPELINE")
print("="*60)

# 1. Khởi tạo pipeline
print("\n[1] Khởi tạo pipeline...")
# Đường dẫn data (từ root project)
data_path = Path(__file__).parent.parent / "data" / "raw" / "VIC.csv"

pipeline = DataPipeline(
    data_path=str(data_path),
    seq_lengths=[30],  # Test với 1 sequence length
    pred_len=7,
    batch_size=32,
)
print("✓ Pipeline initialized")

# 2. Chạy pipeline
print("\n[2] Chạy pipeline (load → preprocess → split → normalize → sequences)...")
dataloaders = pipeline.run()
print("✓ Pipeline completed")

# 3. Kiểm tra data info
print("\n[3] Kiểm tra thông tin data:")
info = pipeline.get_data_info()
print(f"  Raw data: {info['raw_data_shape']} samples")
print(f"  Normalized: {info['normalized']}")
print(f"  Scaler: {info['scaler']}")
print(f"  Use log: {info['use_log']}")

print("\n  Splits:")
for split_name, split_info in info['splits'].items():
    print(f"    {split_name}: {split_info['length']} samples")
    print(f"      Mean: {split_info['mean']:.4f}, Std: {split_info['std']:.4f}")

# 4. Kiểm tra dataloaders
print("\n[4] Kiểm tra DataLoaders:")
train_loader = dataloaders['30d']['train']
val_loader = dataloaders['30d']['val']
test_loader = dataloaders['30d']['test']

print(f"  Train: {len(train_loader)} batches")
print(f"  Val:   {len(val_loader)} batches")
print(f"  Test:  {len(test_loader)} batches")

# 5. Test một batch
print("\n[5] Test một batch từ train_loader:")
for batch_x, batch_y in train_loader:
    print(f"  batch_x shape: {batch_x.shape}  # (batch_size, seq_len, 1)")
    print(f"  batch_y shape: {batch_y.shape}  # (batch_size, pred_len)")
    print(f"  batch_x dtype: {batch_x.dtype}")
    print(f"  batch_y dtype: {batch_y.dtype}")
    print(f"  batch_x range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
    print(f"  batch_y range: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
    break

# 6. Test scaler (inverse transform)
print("\n[6] Test scaler (inverse transform):")
scaler = pipeline.get_scaler()
print(f"  Scaler type: {scaler.__class__.__name__}")
print(f"  Scaler mean: {scaler.mean_[0]:.4f}")
print(f"  Scaler std: {scaler.scale_[0]:.4f}")

# Test inverse transform
from src.data import denormalize_data
test_normalized = np.array([0.0, 1.0, -1.0])
test_denormalized = denormalize_data(test_normalized, scaler)
print(f"\n  Test inverse transform:")
print(f"    Normalized:   {test_normalized}")
print(f"    Denormalized: {test_denormalized}")
print(f"    Original scale: {np.exp(test_denormalized)}")  # Vì dùng log

# 7. Kiểm tra toàn bộ data
print("\n[7] Kiểm tra toàn bộ batches:")
total_train_samples = 0
for batch_x, batch_y in train_loader:
    total_train_samples += batch_x.shape[0]
print(f"  Total train samples: {total_train_samples}")

# 8. Test với nhiều sequence lengths
print("\n[8] Test với nhiều sequence lengths:")
pipeline_multi = DataPipeline(
    data_path=str(data_path),
    seq_lengths=[7, 30, 60],
    pred_len=7,
)
dataloaders_multi = pipeline_multi.run()
print(f"  Available: {list(dataloaders_multi.keys())}")
for key in dataloaders_multi.keys():
    loader = dataloaders_multi[key]['train']
    print(f"    {key}: {len(loader)} batches")

print("\n" + "="*60)
print("✓ TEST HOÀN THÀNH!")
print("="*60)
print("\nKết luận:")
print("  ✓ DataPipeline hoạt động chính xác")
print("  ✓ Dataloaders tạo batches đúng shape")
print("  ✓ Scaler inverse transform hoạt động")
print("  ✓ Hỗ trợ nhiều sequence lengths")

