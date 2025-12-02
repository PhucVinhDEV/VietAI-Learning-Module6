"""Script để train FPT GRU model và lưu checkpoint."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data import load_fpt_data, prepare_data, TimeSeriesDataset
from src.model import GRUModel
from src.training import train_model
from src.utils import CONFIG, SEED, set_seed, evaluate_model
from src.utils.checkpoint import save_checkpoint


def main():
    """Train model và save checkpoint."""
    # Set random seed để đảm bảo reproducibility
    set_seed(SEED)
    print(f"🔒 Random seed set to: {SEED}")
    
    print("=" * 60)
    print("FPT GRU Model Training")
    print("=" * 60)
    
    # Load và prepare data
    print("\n📊 Loading data...")
    df = load_fpt_data()
    df_processed = prepare_data(df)
    print(f"✅ Data loaded: {len(df_processed)} records")
    
    # Use default config hoặc có thể override
    config = CONFIG.copy()
    print(f"\n⚙️  Config: {config}")
    
    # Split data
    input_len = config["input_len"]
    output_len = config["output_len"]
    val_size = config["val_size"]
    
    val_window_len = val_size + input_len + output_len - 1
    df_train = df_processed.iloc[:-val_window_len].reset_index(drop=True)
    df_val = df_processed.iloc[-val_window_len:].reset_index(drop=True)
    
    print(f"📈 Train: {len(df_train)} records, Val: {len(df_val)} records")
    
    # Create datasets
    feature_cols = ["close_log"]
    print("\n🔧 Creating datasets...")
    train_ds = TimeSeriesDataset(
        df_train, feature_cols, input_len, output_len,
        scaler=None, fit_scaler=True
    )
    val_ds = TimeSeriesDataset(
        df_val, feature_cols, input_len, output_len,
        scaler=train_ds.scaler
    )
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    
    print(f"✅ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = GRUModel(
        input_size=len(feature_cols),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_len=output_len
    ).to(config["device"])
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    print("\n🚀 Training model...")
    best_state, train_losses, val_losses = train_model(
        model, train_loader, val_loader, config, verbose=True
    )
    
    # Load best state
    model.load_state_dict(best_state)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    mape, preds, true = evaluate_model(model, val_loader, config["device"])
    
    metrics = {
        'mape': float(mape),
        'best_val_loss': float(min(val_losses)),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
    }
    
    print(f"✅ Validation MAPE: {mape:.2f}%")
    print(f"✅ Best Val Loss: {min(val_losses):.6f}")
    
    # Save checkpoint
    checkpoint_dir = project_root / "models" / "fpt_gru"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    print(f"\n💾 Saving checkpoint to: {checkpoint_path}")
    save_checkpoint(
        model_state_dict=best_state,
        config=config,
        scaler=train_ds.scaler,
        metrics=metrics,
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=checkpoint_path
    )
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"📁 Checkpoint saved: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

