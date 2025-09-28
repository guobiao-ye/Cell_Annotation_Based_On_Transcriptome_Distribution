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

# --- 1. Custom Dataset Class (CellGeneDataset) ---
class CellGeneDataset(Dataset):
    """
    Custom PyTorch dataset for loading cell gene [3D] point cloud data.
    """
    def __init__(self, csv_file, class_to_idx, gene_to_idx):
        self.df = pd.read_csv(csv_file)
        self.class_to_idx = class_to_idx
        self.gene_to_idx = gene_to_idx
        self.num_genes = len(gene_to_idx)
        
        self.grouped = self.df.groupby('cell_id')
        self.cell_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        cell_data = self.grouped.get_group(cell_id)
        

        coords = cell_data[['normalized_x', 'normalized_y', 'z']].values
        
        # Extract gene IDs and perform one-hot encoding
        gene_indices = cell_data['target_gene'].map(self.gene_to_idx).values
        one_hot_genes = np.zeros((len(gene_indices), self.num_genes), dtype=np.float32)
        one_hot_genes[np.arange(len(gene_indices)), gene_indices] = 1.0
        

        features = np.hstack([coords, one_hot_genes]).astype(np.float32)
        
        # Get the label
        label_str = cell_data['class'].iloc[0]
        label = self.class_to_idx[label_str]
        
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
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_features, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        point_features = self.feature_extractor(x)
        global_feature, _ = torch.max(point_features, 2)
        logits = self.classifier(global_feature)
        return logits

# --- 4. Training and Validation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()

# --- 5. Visualization Function ---
def plot_history(history, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True)
    ax3.plot(history['lr'], label='Learning Rate', color='orange')
    ax3.set_title('Learning Rate Schedule'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log'); ax3.legend(); ax3.grid(True)
    plt.tight_layout(); plt.savefig(save_path)
    print(f"\nTraining history plot saved to: {save_path}")

# --- 6. Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train a [3D] point cloud model to classify cell types.")
    parser.add_argument('--train_csv', required=True, help='Path to the training set CSV file')
    parser.add_argument('--test_csv', required=True, help='Path to the test set CSV file')
    parser.add_argument('--color_map', required=True, help='Path to the gene color map file')
    parser.add_argument('--output_dir', default='./model_output_3d', help='Directory to save the model and results')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lr_patience', type=int, default=3, help='Patience for the learning rate scheduler (epochs)')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate decay factor')
    args = parser.parse_args()

    # ... (Directory and log file setup remains the same) ...
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / 'training_log.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data Preparation ---
    print("\nStep 1/4: Preparing [3D] data loaders...")
    gene_map_df = pd.read_csv(args.color_map)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_map_df['gene'])}
    num_genes = len(gene_to_idx)
    train_df_for_meta = pd.read_csv(args.train_csv)
    # Check if 'z' column exists
    if 'z' not in train_df_for_meta.columns:
        print("Error: 'z' column not found in the input CSV file. Please check the data.")
        return
    unique_classes = sorted(train_df_for_meta['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    num_classes = len(class_to_idx)
    train_dataset = CellGeneDataset(args.train_csv, class_to_idx, gene_to_idx)
    test_dataset = CellGeneDataset(args.test_csv, class_to_idx, gene_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Data preparation complete. Found {num_classes} cell classes and {num_genes} types of genes.")
    
    # --- Model Preparation ---
    print("\nStep 2/4: Building the model...")
    num_features = 3 + num_genes
    print(f"The model will use {num_features} input features (3 for coords + {num_genes} for one-hot genes).")
    
    model = PointNetClassifier(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True)
    
    # --- Training Loop ---
    print("\nStep 3/4: Starting training...")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"--- Training started (3D mode): {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}\n")
        log_file.write(f"Scheduler: ReduceLROnPlateau, patience={args.lr_patience}, factor={args.lr_factor}\n")
        log_file.write("-" * 50 + "\n")
        print(f"Training log will be saved to: {log_file_path}")
    
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            log_str = (
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            print(log_str)
            with open(log_file_path, 'a') as f:
                f.write(log_str + '\n')
            
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            torch.save(model.state_dict(), output_dir / f"model_epoch_{epoch+1}.pth")
        
        with open(log_file_path, 'a') as f:
            f.write("-" * 50 + "\n"); f.write(f"--- Training finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    print("\nTraining complete!")
    
    # --- Results Visualization ---
    print("\nStep 4/4: Generating results visualization...")
    plot_history(history, output_dir / 'training_history.png')
    
if __name__ == '__main__':
    main()