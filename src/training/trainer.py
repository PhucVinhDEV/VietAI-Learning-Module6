"""Training utilities cho model."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_loader, val_loader, config, verbose=True):
    """
    Train model với early stopping.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader cho training
        val_loader: DataLoader cho validation
        config: Config dict với training parameters
        verbose: In progress nếu True
    
    Returns:
        best_state_dict: Best model state dict
        train_losses: List training losses
        val_losses: List validation losses
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = config["device"]
    
    best_loss = float('inf')
    best_state = None
    patience = config["early_stop_patience"]
    no_up = 0
    train_losses = []
    val_losses = []
    
    epochs = range(config["num_epochs"])
    if verbose:
        epochs = tqdm(epochs, desc="Training")
    
    for epoch in epochs:
        # Training phase
        model.train()
        tl = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            pred = model(xb).squeeze()
            yb = yb.squeeze()
            
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tl.append(loss.item())
        
        # Validation phase
        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                yb = yb.squeeze()
                vl.append(criterion(pred, yb).item())
        
        tl_mean = np.mean(tl)
        vl_mean = np.mean(vl)
        train_losses.append(tl_mean)
        val_losses.append(vl_mean)
        
        if verbose:
            epochs.set_postfix({
                'train_loss': f'{tl_mean:.6f}',
                'val_loss': f'{vl_mean:.6f}'
            })
        
        # Early stopping
        if vl_mean < best_loss:
            best_loss = vl_mean
            best_state = model.state_dict()
            no_up = 0
        else:
            no_up += 1
            if no_up >= patience:
                if verbose:
                    print(f"\nEarly stopping tại epoch {epoch+1}!")
                break
    
    return best_state, train_losses, val_losses

