import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# --- 1. Custom Dataset Class (CellGeneDataset) - Core Modification ---
class CellGeneDataset(Dataset):
    """
    [Ablation Study V2] Custom PyTorch dataset that only loads spatial coordinates, ignoring gene category information.
    """
    def __init__(self, csv_file, class_to_idx):
        self.df = pd.read_csv(csv_file)
        self.class_to_idx = class_to_idx
        
        self.grouped = self.df.groupby('cell_id')
        self.cell_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        cell_data = self.grouped.get_group(cell_id)
        
        # Only extract x, y, z three coordinates as features
        coords = cell_data[['normalized_x', 'normalized_y', 'z']].values
        features = coords.astype(np.float32)
        
        # Get the label
        label_str = cell_data['class'].iloc[0]
        label = self.class_to_idx[str(label_str)]
        
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

# --- 2. Custom Data Loading Function (collate_fn) ---

def collate_fn(batch):
    features_list, labels_list = zip(*batch)
    max_len = max(len(f) for f in features_list)
    num_features = features_list[0].shape[1]
    padded_features = torch.zeros(len(batch), max_len, num_features)
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len, :] = features
    labels = torch.stack(labels_list)
    return padded_features, labels

# --- 3. Model Definition (PointNet-like Classifier) ---

class PointNetClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(PointNetClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Conv1d(num_features, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                               nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                               nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                          nn.Linear(256, num_classes))
    def forward(self, x):
        x = x.transpose(1, 2); point_features = self.feature_extractor(x)
        global_feature, _ = torch.max(point_features, 2); logits = self.classifier(global_feature)
        return logits

# --- 4. Training and Validation Functions ---

def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train(); running_loss = 0.0; correct_predictions = 0; total_samples = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
        outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        running_loss += loss.item() * inputs.size(0); _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data); total_samples += inputs.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

def validate_epoch(model, dataloader, criterion, device):

    model.eval(); running_loss = 0.0; correct_predictions = 0; total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0); _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data); total_samples += inputs.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

# --- 5. Visualization Function ---

def plot_history(history, save_path):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    ax1.plot(history['train_acc'], label='Train Accuracy'); ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_loss'], label='Train Loss'); ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss'); ax2.legend(); ax2.grid(True)
    ax3.plot(history['lr'], label='Learning Rate', color='orange'); ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log'); ax3.legend(); ax3.grid(True); plt.tight_layout(); plt.savefig(save_path)
    print(f"\nTraining history plot saved to: {save_path}")

# --- 6. Main Function ---
def main():
    parser = argparse.ArgumentParser(description="[Ablation Study V2] Train a point cloud model to classify cell types using only spatial coordinate information.")
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--output_dir', default='./model_output_NO_GENES', help='Directory to save the model and results')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / 'training_log.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nStep 1/4: Preparing [gene-free] data loaders...")
    train_df_for_meta = pd.read_csv(args.train_csv)
    unique_classes = sorted(train_df_for_meta['class'].astype(str).unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    num_classes = len(class_to_idx)
    # The dataset class no longer needs gene_to_idx
    train_dataset = CellGeneDataset(args.train_csv, class_to_idx)
    test_dataset = CellGeneDataset(args.test_csv, class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Data preparation complete.")
    
    print("\nStep 2/4: Building the model...")
    num_features = 3 # The number of features is now fixed to 3 (x, y, z)
    print(f"The model will use {num_features} input features (spatial coordinates only).")
    
    model = PointNetClassifier(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True)
    
    print("\nStep 3/4: Starting training (Ablation Study V2)...")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"--- Training started (Ablation study without gene information): {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.write(f"Device: {device}\n"); log_file.write("-" * 50 + "\n")
        print(f"Training log will be saved to: {log_file_path}")
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
            scheduler.step(val_loss); current_lr = optimizer.param_groups[0]['lr']
            log_str = (f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            print(log_str); log_file.write(log_str + '\n')
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch+1}.pth")
    
    print("\nTraining complete!")
    
    print("\nStep 4/4: Generating results visualization...")
    plot_history(history, output_dir / 'training_history.png')

if __name__ == '__main__':
    main()