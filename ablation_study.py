import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import json
import warnings
import gc

warnings.filterwarnings('ignore')

# Memory management tools
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

# SimpleAttention module
class SimpleAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleAttention, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, input_dim = x.shape
        attn_weights = self.attention_weights(x)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            attn_weights = attn_weights * mask
            attn_sum = attn_weights.sum(dim=1, keepdim=True) + 1e-8
            attn_weights = attn_weights / attn_sum
        transformed_features = self.feature_transform(x)
        attended_features = (transformed_features * attn_weights).sum(dim=1)
        return attended_features

# SimpleMemoryDataset
class SimpleMemoryDataset(Dataset):
    def __init__(self, csv_file, class_to_idx, gene_to_idx, max_points_per_cell=400, spatial_normalize=True):
        self.df = pd.read_csv(csv_file)
        self.class_to_idx = class_to_idx
        self.gene_to_idx = gene_to_idx
        self.num_genes = len(gene_to_idx)
        self.max_points_per_cell = max_points_per_cell
        if spatial_normalize:
            for col in ['normalized_x', 'normalized_y', 'z']:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std() + 1e-8
                self.df[col] = (self.df[col] - mean_val) / std_val
        self.df = self.df.fillna(0)
        self.grouped = self.df.groupby('cell_id')
        self.cell_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        cell_data = self.grouped.get_group(cell_id)
        coords = cell_data[['normalized_x', 'normalized_y', 'z']].values.astype(np.float32)
        gene_indices = cell_data['target_gene'].map(self.gene_to_idx).values.astype(np.int32)
        if len(coords) > self.max_points_per_cell:
            indices = np.random.choice(len(coords), self.max_points_per_cell, replace=False)
            coords = coords[indices]
            gene_indices = gene_indices[indices]
        label_str = cell_data['class'].iloc[0]
        label = self.class_to_idx[label_str]
        return {
            'coords': torch.from_numpy(coords),
            'gene_indices': torch.from_numpy(gene_indices),
            'label': torch.tensor(label, dtype=torch.long),
            'num_points': len(coords)
        }

# simple_collate_fn
def simple_collate_fn(batch):
    max_len = max(item['num_points'] for item in batch)
    batch_size = len(batch)
    coords_batch = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    gene_indices_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels_batch = torch.zeros(batch_size, dtype=torch.long)
    masks_batch = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        seq_len = item['num_points']
        coords_batch[i, :seq_len] = item['coords']
        gene_indices_batch[i, :seq_len] = item['gene_indices']
        labels_batch[i] = item['label']
        masks_batch[i, :seq_len] = True
    return {
        'coords': coords_batch,
        'gene_indices': gene_indices_batch,
        'labels': labels_batch,
        'masks': masks_batch
    }

# SimpleTrainer class (copied from original script)
class SimpleTrainer:
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            coords = batch['coords'].to(self.device, non_blocking=True)
            gene_indices = batch['gene_indices'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            masks = batch['masks'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = self.model(coords, gene_indices, masks)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * coords.size(0)
            _, preds = torch.max(logits, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += coords.size(0)
            
            if batch_idx % 50 == 0:
                clear_memory()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                coords = batch['coords'].to(self.device, non_blocking=True)
                gene_indices = batch['gene_indices'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                masks = batch['masks'].to(self.device, non_blocking=True)
                
                logits = self.model(coords, gene_indices, masks)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item() * coords.size(0)
                _, preds = torch.max(logits, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += coords.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        
        return epoch_loss, epoch_acc.item(), all_preds, all_labels

# AblationClassifier
class AblationClassifier(nn.Module):
    def __init__(self, num_genes, num_classes, hidden_dim=128, gene_embed_dim=32, dropout_rate=0.2,
                 use_spatial=True, use_gene=True):
        super(AblationClassifier, self).__init__()
        self.use_spatial = use_spatial
        self.use_gene = use_gene
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.gene_embed_dim = gene_embed_dim
        
        if self.use_gene:
            self.gene_embedding = nn.Embedding(num_genes, gene_embed_dim)
            self.gene_encoder = nn.Sequential(
                nn.Linear(gene_embed_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(dropout_rate)
            )
            self.gene_attention = SimpleAttention(hidden_dim // 2, hidden_dim // 4)
        else:
            self.gene_embedding = None
            self.gene_encoder = None
            self.gene_attention = None
        
        if self.use_spatial:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(dropout_rate)
            )
            self.spatial_attention = SimpleAttention(hidden_dim // 2, hidden_dim // 4)
        else:
            self.spatial_encoder = None
            self.spatial_attention = None
        
        input_dim = (hidden_dim // 4) * (int(use_spatial) + int(use_gene))
        if input_dim == 0:
            input_dim = hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, coords, gene_indices, masks=None):
        batch_size, seq_len = coords.shape[:2]
        features = []
        
        if self.use_spatial:
            coords_flat = coords.view(-1, 3)
            spatial_features_flat = self.spatial_encoder(coords_flat)
            spatial_features = spatial_features_flat.view(batch_size, seq_len, -1)
            spatial_global = self.spatial_attention(spatial_features, masks)
            features.append(spatial_global)
        
        if self.use_gene:
            gene_embeds = self.gene_embedding(gene_indices)
            gene_embeds_flat = gene_embeds.view(-1, self.gene_embed_dim)
            gene_features_flat = self.gene_encoder(gene_embeds_flat)
            gene_features = gene_features_flat.view(batch_size, seq_len, -1)
            gene_global = self.gene_attention(gene_features, masks)
            features.append(gene_global)
        
        if not features:
            dummy_features = torch.zeros(batch_size, self.hidden_dim // 2, device=coords.device)
            features.append(dummy_features)
        
        combined_features = torch.cat(features, dim=1)
        logits = self.classifier(combined_features)
        return logits

# AblationTrainer
class AblationTrainer(SimpleTrainer):
    def __init__(self, model, device, output_dir, experiment_name):
        super().__init__(model, device, output_dir)
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Ablation study for 3D cell classifier")
    parser.add_argument('--train_csv', required=True, help='Path to the training set CSV file')
    parser.add_argument('--test_csv', required=True, help='Path to the test set CSV file')
    parser.add_argument('--color_map', required=True, help='Path to the gene color map file')
    parser.add_argument('--output_dir', default='./ablation_results', help='Directory to save ablation study results')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--gene_embed_dim', type=int, default=32, help='Gene embedding dimension')
    parser.add_argument('--max_points', type=int, default=400, help='Maximum number of points per cell')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--spatial_normalize', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    clear_memory()
    
    # Data preparation
    print("\n=== Data Preparation ===")
    gene_map_df = pd.read_csv(args.color_map)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_map_df['gene'])}
    num_genes = len(gene_to_idx)
    train_df_for_meta = pd.read_csv(args.train_csv)
    unique_classes = sorted(train_df_for_meta['class'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    num_classes = len(class_to_idx)
    
    train_dataset = SimpleMemoryDataset(
        args.train_csv, class_to_idx, gene_to_idx,
        max_points_per_cell=args.max_points,
        spatial_normalize=args.spatial_normalize
    )
    test_dataset = SimpleMemoryDataset(
        args.test_csv, class_to_idx, gene_to_idx,
        max_points_per_cell=args.max_points,
        spatial_normalize=args.spatial_normalize
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=simple_collate_fn, num_workers=2, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=simple_collate_fn, num_workers=2, pin_memory=False
    )
    
    print(f"Data preparation complete: {num_classes} classes, {num_genes} genes, {len(train_dataset)} training samples")
    
    # Ablation study configuration
    ablation_configs = [
        {'name': 'full_model', 'use_spatial': True, 'use_gene': True},
        {'name': 'spatial_only', 'use_spatial': True, 'use_gene': False},
        {'name': 'gene_only', 'use_spatial': False, 'use_gene': True},
        {'name': 'random_baseline', 'use_spatial': False, 'use_gene': False}
    ]
    
    results = {}
    
    for config in ablation_configs:
        print(f"\n=== Running Ablation Experiment: {config['name']} ===")
        model = AblationClassifier(
            num_genes=num_genes,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            gene_embed_dim=args.gene_embed_dim,
            dropout_rate=args.dropout_rate,
            use_spatial=config['use_spatial'],
            use_gene=config['use_gene']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}, Size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        trainer = AblationTrainer(model, device, args.output_dir, config['name'])
        
        best_val_acc = 0.0
        patience = 0
        
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ({config['name']}) ---")
            allocated, reserved = get_memory_usage()
            print(f"GPU Memory: {allocated:.2f} GB")
            
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer)
            val_loss, val_acc, val_preds, val_labels = trainer.validate_epoch(test_loader)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            trainer.history['train_loss'].append(train_loss)
            trainer.history['train_acc'].append(train_acc)
            trainer.history['val_loss'].append(val_loss)
            trainer.history['val_acc'].append(val_acc)
            trainer.history['lr'].append(current_lr)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_to_idx': class_to_idx,
                    'gene_to_idx': gene_to_idx,
                }, trainer.output_dir / 'best_model.pth')
            else:
                patience += 1
            
            print(f"Train: {train_loss:.4f}/{train_acc:.4f}, Val: {val_loss:.4f}/{val_acc:.4f}, "
                  f"Best: {best_val_acc:.4f}, LR: {current_lr:.6f}")
            
            clear_memory()
            
            if patience >= 8:
                print("Early stopping")
                break
        
        # Final evaluation
        checkpoint = torch.load(trainer.output_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        final_val_loss, final_val_acc, final_preds, final_labels = trainer.validate_epoch(test_loader)
        
        class_names = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        report = classification_report(final_labels, final_preds, target_names=class_names, output_dict=True)
        
        results[config['name']] = {
            'val_acc': final_val_acc,
            'macro_f1': report['macro avg']['f1-score'],
            'history': trainer.history
        }
        
        with open(trainer.output_dir / 'results.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.history['train_acc'], label='Train')
        plt.plot(trainer.history['val_acc'], label='Validation')
        plt.title(f'Accuracy ({config["name"]})')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(trainer.history['train_loss'], label='Train')
        plt.plot(trainer.history['val_loss'], label='Validation')
        plt.title(f'Loss ({config["name"]})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(trainer.output_dir / 'training_history.png', dpi=300)
        plt.close()
    
    # Comparative visualization
    plt.figure(figsize=(12, 6))
    names = [config['name'] for config in ablation_configs]
    val_accs = [results[name]['val_acc'] for name in names]
    macro_f1s = [results[name]['macro_f1'] for name in names]
    
    plt.subplot(1, 2, 1)
    plt.bar(names, val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Validation Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar(names, macro_f1s, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Macro F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Macro F1')
    
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / 'ablation_comparison.png', dpi=300)
    plt.close()
    
    # Save ablation study results
    with open(Path(args.output_dir) / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Ablation Study Complete ===")
    for name in results:
        print(f"{name}: Validation Accuracy={results[name]['val_acc']:.4f}, Macro F1={results[name]['macro_f1']:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()