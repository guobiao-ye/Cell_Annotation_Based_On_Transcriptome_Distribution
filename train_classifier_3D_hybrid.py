import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import warnings
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import random
import math

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FastCellGeneDataset(Dataset):
    """
    Fast-converging cell gene 3D point cloud dataset - mimics the simple and efficient approach of v1.
    """
    def __init__(self, csv_file, class_to_idx, gene_to_idx, 
                 augment=False, max_points=800):
        self.df = pd.read_csv(csv_file)
        self.class_to_idx = class_to_idx
        self.gene_to_idx = gene_to_idx
        self.num_genes = len(gene_to_idx)
        self.augment = augment
        self.max_points = max_points
        
        # Simple grouping, similar to v1
        self.grouped = self.df.groupby('cell_id')
        self.cell_ids = list(self.grouped.groups.keys())
        
        # Filter invalid cells
        valid_cells = []
        for cell_id in self.cell_ids:
            cell_data = self.grouped.get_group(cell_id)
            if len(cell_data) >= 10:
                valid_cells.append(cell_id)
        self.cell_ids = valid_cells
        print(f"Number of valid cells: {len(self.cell_ids)}")
    
    def __len__(self):
        return len(self.cell_ids)
    
    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        cell_data = self.grouped.get_group(cell_id)
        
        # Extract 3D coordinates
        coords = cell_data[['normalized_x', 'normalized_y', 'z']].values.astype(np.float32)
        
        # Extract genes and perform one-hot encoding (similar to v1)
        gene_indices = cell_data['target_gene'].map(self.gene_to_idx).values
        one_hot_genes = np.zeros((len(gene_indices), self.num_genes), dtype=np.float32)
        one_hot_genes[np.arange(len(gene_indices)), gene_indices] = 1.0
        
        # Combine features (exactly mimicking the v1 approach)
        features = np.hstack([coords, one_hot_genes]).astype(np.float32)
        
        # Point number limit
        if len(features) > self.max_points:
            indices = np.random.choice(len(features), self.max_points, replace=False)
            features = features[indices]
        
        # Simple data augmentation
        if self.augment and np.random.random() < 0.3:
            # Simple noise augmentation only
            noise = np.random.normal(0, 0.01, features[:, :3].shape)
            features[:, :3] += noise
        
        # Get label
        label_str = cell_data['class'].iloc[0]
        label = self.class_to_idx[label_str]
        
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """Identical collate function to v1."""
    features_list, labels_list = zip(*batch)
    max_len = max(len(f) for f in features_list)
    num_features = features_list[0].shape[1]
    padded_features = torch.zeros(len(batch), max_len, num_features)
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len, :] = features
    labels = torch.stack(labels_list)
    return padded_features, labels

class HybridPointNetClassifier(nn.Module):
    """
    Hybrid architecture: Fast-converging CNN backbone + lightweight Transformer enhancement.
    """
    def __init__(self, num_features, num_classes, use_attention=True, d_model=256):
        super().__init__()
        self.use_attention = use_attention
        
        # === v1-style CNN backbone (ensures fast convergence) ===
        self.cnn_backbone = nn.Sequential(
            # First layer: Fast feature extraction
            nn.Conv1d(num_features, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Second layer: Intermediate features
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Third layer: High-level features
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # === Optional lightweight attention enhancement ===
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=256, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(256)
        
        # === v1-style classification head (ensures stable training) ===
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights (important!)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Good weight initialization to accelerate convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size, seq_len, num_features = x.shape
        
        # === CNN feature extraction (like v1, fast and effective) ===
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        cnn_features = self.cnn_backbone(x_cnn)  # [batch, 256, seq_len]
        cnn_features = cnn_features.transpose(1, 2)  # [batch, seq_len, 256]
        
        # === Optional attention enhancement ===
        if self.use_attention:
            # Self-attention
            attn_out, _ = self.attention(cnn_features, cnn_features, cnn_features)
            # Residual connection
            enhanced_features = self.attention_norm(cnn_features + attn_out)
        else:
            enhanced_features = cnn_features
        
        # === Global feature aggregation (v1-style max pooling) ===
        global_feature, _ = torch.max(enhanced_features, dim=1)  # [batch, 256]
        
        # === Classification ===
        logits = self.classifier(global_feature)
        
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Identical training function to v1 for consistency."""
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
    """Identical validation function to v1."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item(), all_preds, all_labels

def plot_history(history, save_path):
    """Improved visualization function."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(history['lr'], label='Learning Rate', color='orange', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1' in history:
        axes[1, 1].plot(history['val_f1'], label='Validation F1', color='green', linewidth=2)
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Fast-converging hybrid 3D cell gene point cloud classifier.")
    
    # Data parameters
    parser.add_argument('--train_csv', required=True, help='Path to the training set CSV file')
    parser.add_argument('--test_csv', required=True, help='Path to the test set CSV file')
    parser.add_argument('--color_map', required=True, help='Path to the gene color map file')
    parser.add_argument('--output_dir', default='./hybrid_output', help='Output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr_patience', type=int, default=3, help='Patience for the learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Learning rate decay factor')
    
    # Model parameters
    parser.add_argument('--use_attention', action='store_true', help='Use attention enhancement')
    parser.add_argument('--max_points', type=int, default=800, help='Maximum number of points per cell')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / 'training_log.txt'
    
    # --- Data preparation (exactly mimicking the v1 approach) ---
    print("\n=== Step 1/4: Data Preparation ===")
    
    # Load gene map
    gene_map_df = pd.read_csv(args.color_map)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_map_df['gene'])}
    num_genes = len(gene_to_idx)
    
    # Build class map
    train_df_for_meta = pd.read_csv(args.train_csv)
    if 'z' not in train_df_for_meta.columns:
        print("Error: 'z' column not found in the input CSV file. Please check the data.")
        return
    unique_classes = sorted(train_df_for_meta['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    num_classes = len(class_to_idx)
    
    print(f"Found {num_classes} cell classes: {unique_classes}")
    print(f"Found {num_genes} types of genes")
    
    # Create datasets
    train_dataset = FastCellGeneDataset(
        args.train_csv, class_to_idx, gene_to_idx,
        augment=True, max_points=args.max_points
    )
    
    test_dataset = FastCellGeneDataset(
        args.test_csv, class_to_idx, gene_to_idx,
        augment=False, max_points=args.max_points
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Training set: {len(train_dataset)} cells")
    print(f"Test set: {len(test_dataset)} cells")
    
    # --- Model building ---
    print("\n=== Step 2/4: Model Building ===")
    
    # Calculate number of features (3D coordinates + one-hot genes)
    num_features = 3 + num_genes
    print(f"Input feature dimension: {num_features} (3D coordinates + {num_genes} genes)")
    
    model = HybridPointNetClassifier(
        num_features=num_features,
        num_classes=num_classes,
        use_attention=args.use_attention
    ).to(device)
    
    # Calculate model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params:,}")
    print(f"Using attention enhancement: {args.use_attention}")
    
    # --- Training setup ---
    print("\n=== Step 3/4: Training Setup ===")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, 
                                 patience=args.lr_patience, verbose=True)
    
    # --- Training loop ---
    print(f"\n=== Step 4/4: Starting training (for {args.epochs} epochs) ===")
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    best_val_acc = 0.0
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=== Fast-converging Hybrid 3D Cell Gene Point Cloud Classifier Training Log ===\n")
        log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Model parameters: {total_params:,}\n")
        log_file.write(f"Using attention: {args.use_attention}\n")
        log_file.write(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}\n")
        log_file.write("-" * 80 + "\n")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, test_loader, criterion, device)
        
        # Calculate F1 score
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Log record
        log_str = (
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | Learning Rate: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_str + '\n')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_to_idx': class_to_idx,
                'gene_to_idx': gene_to_idx,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            print(f"  → New best model saved! Accuracy: {val_acc:.4f}")
    
    # Training complete
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("-" * 80 + "\n")
        f.write(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    
    # --- Final evaluation ---
    print("\n=== Generating Analysis Report ===")
    
    # Load the best model for evaluation
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_loss, final_val_acc, final_preds, final_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"Final validation results - Accuracy: {final_val_acc:.4f}, Loss: {final_val_loss:.4f}")
    
    # Generate classification report
    report = classification_report(final_labels, final_preds, 
                                 target_names=unique_classes, 
                                 output_dict=True)
    
    with open(output_dir / 'classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Plot results
    plot_history(history, output_dir / 'training_history.png')
    plot_confusion_matrix(final_labels, final_preds, unique_classes, 
                        output_dir / 'confusion_matrix.png')
    
    # Save configuration and history
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    
    config = {
        'model_config': {
            'num_classes': num_classes,
            'num_genes': num_genes,
            'num_features': num_features,
            'use_attention': args.use_attention,
            'total_params': total_params
        },
        'training_config': vars(args),
        'results': {
            'best_val_acc': best_val_acc,
            'final_val_acc': final_val_acc,
            'total_epochs': args.epochs
        }
    }
    
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll results have been saved to: {output_dir}")
    print("Training finished!")

if __name__ == '__main__':
    main()