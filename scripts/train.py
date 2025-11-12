import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import os
import argparse
from tqdm import tqdm
import sys

# Add the 'src' directory to the Python path
# This allows us to import from 'src.model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import UNet1D

class GWDataset(Dataset):
    """Custom Dataset for loading GW .npy data."""
    def __init__(self, signals_path, masks_path):
        self.signals = np.load(signals_path)
        self.masks = np.load(masks_path)
        print(f"Loaded signals with shape: {self.signals.shape}")
        print(f"Loaded masks with shape: {self.masks.shape}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        mask = self.masks[idx]

        # Add a channel dimension: (Length,) -> (1, Length)
        # PyTorch Conv1d expects (Batch, Channels, Length)
        return torch.from_numpy(signal).float().unsqueeze(0), \
               torch.from_numpy(mask).float().unsqueeze(0)


def calculate_f1(y_pred, y_true, threshold=0.5):
    """Calculates F1 score for a batch of predictions."""
    # Apply sigmoid and threshold to get binary predictions
    preds = torch.sigmoid(y_pred) > threshold

    # Move to CPU and flatten
    preds = preds.cpu().numpy().flatten()
    y_true = y_true.cpu().numpy().flatten()

    # Calculate F1 score
    return f1_score(y_true, preds, zero_division=0)


def main(args):
    """Main training and validation loop."""

    # 1. Setup device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    signals_file = os.path.join(args.data_dir, "noisy_signals.npy")
    masks_file = os.path.join(args.data_dir, "masks.npy")

    if not os.path.exists(signals_file) or not os.path.exists(masks_file):
        print(f"Error: Data files not found in {args.data_dir}")
        print(f"Please run 'python scripts/data_generator.py --output-dir {args.data_dir}' first.")
        return

    dataset = GWDataset(signals_file, masks_file)

    # 3. Split Data (80% train, 20% validation)
    val_percent = 0.2
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], 
                                    generator=torch.Generator().manual_seed(42))

    # 4. Create DataLoaders
    # Set num_workers > 0 for multi-process data loading
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training on {n_train} samples, validating on {n_val} samples.")

    # 5. Initialize Model, Loss, Optimizer
    model = UNet1D(n_channels=1, n_classes=1).to(device)

    # Use BCEWithLogitsLoss: it combines Sigmoid + BCELoss for better
    # numerical stability. It expects raw logits from the model.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 6. Training Loop
    best_val_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0
    
        # Use tqdm for a progress bar
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='sample') as pbar:
            for signals, masks in train_loader:
                # Move data to the configured device
                signals = signals.to(device=device)
                masks = masks.to(device=device)

                # Forward pass
                outputs = model(signals)
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(signals.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        train_loss = epoch_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_f1 = 0
        with torch.no_grad(): # Disable gradient calculation
            for signals, masks in val_loader:
                signals = signals.to(device=device)
                masks = masks.to(device=device)

                outputs = model(signals)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                val_f1 += calculate_f1(outputs, masks)

        val_loss /= len(val_loader)
        val_f1 /= len(val_loader)

        print(f"\nEpoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Save the model if it has the best F1 score so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.model_path)
            print(f"New best model saved to {args.model_path} (F1: {best_val_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Final model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D U-Net for GW signal segmentation.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing .npy files.")
    parser.add_argument("--model-path", type=str, default="unet_gw_model.pth", help="Path to save the trained model (in the root folder).")
    args = parser.parse_args()
    main(args)