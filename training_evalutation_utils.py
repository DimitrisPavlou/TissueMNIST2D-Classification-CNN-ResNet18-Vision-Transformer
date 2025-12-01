import torch 
import torch.nn.functional as F
import numpy as np
#training functions

def train_step(model, dataloader, loss_fn, optimizer, device, scaler=None):
    """
    Perform one training epoch with optional mixed precision training.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        scaler: GradScaler for mixed precision (optional)
    """
    model.train()
    running_loss = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Fix label format
        if y.ndim > 1:
            y = y.squeeze()
        if y.ndim > 1:
            y = y.argmax(dim=1)
        y = y.long()
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(X)
                loss = loss_fn(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * X.size(0)
    
    return running_loss / len(dataloader.dataset)


def validation_step(model, dataloader, loss_fn, device):
    """
    Evaluate model on validation set.
    """
    model.eval()
    running_loss = 0
    correct, total = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Fix label format
            if y.ndim > 1:
                y = y.squeeze()
            if y.ndim > 1:
                y = y.argmax(dim=1)
            y = y.long()
            
            # Forward pass
            logits = model(X)
            loss = loss_fn(logits, y)
            running_loss += loss.item() * X.size(0)
            
            # Calculate accuracy
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, device, 
        early_stopping=False, patience=5, model_name="Model", save_best=True, 
        checkpoint_path=None, use_mixed_precision=True):
    """
    Train model with optional early stopping, model checkpointing, and mixed precision.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Device (CPU/GPU)
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        model_name: Name of the model for display
        save_best: Whether to save the best model
        checkpoint_path: Path to save model checkpoints (default: f"{model_name}_best.pth")
        use_mixed_precision: Whether to use mixed precision training (faster on GPU)
    
    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    
    # Set default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = f"{model_name.replace(' ', '_').lower()}_best.pth"
    
    # Initialize GradScaler for mixed precision training (only on CUDA)
    scaler = None
    if use_mixed_precision and device.type == 'cuda':
        scaler = torch.amp.GradScaler()
        print(f"✓ Mixed precision training enabled")
    elif use_mixed_precision and device.type == 'cpu':
        print(f"⚠ Mixed precision not available on CPU, using standard precision")
    
    print(f"\nTraining {model_name}...")
    print("=" * 70)
    
    for epoch in range(epochs):
        # Training step with optional mixed precision
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device, scaler)
        val_loss, val_acc = validation_step(model, val_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Early stopping logic
        if early_stopping:
            if val_acc > best_val_acc:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    break
                    
        # Save best model
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
        
        
    
    # Load best model if checkpointing was enabled
    if save_best:
        print(f"\n✓ Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Best model from epoch {checkpoint['epoch']} "
              f"(Val Acc: {checkpoint['val_acc']:.4f})")
    
    print("=" * 70)
    return train_losses, val_losses, val_accuracies


def evaluate(model, dataloader, device):
    """
    Comprehensive evaluation with predictions and probabilities.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Fix label format
            if y.ndim > 1:
                y = y.squeeze()
            if y.ndim > 1:
                y = y.argmax(dim=1)
            y = y.long()
            
            # Get predictions and probabilities
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    
    return accuracy, all_preds, all_labels, all_probs