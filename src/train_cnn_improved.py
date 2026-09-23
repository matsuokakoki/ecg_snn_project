"""
Improved CNN baseline training with proper hyperparameter tuning.
Goal: Achieve Acc >= 0.90 for fair comparison.
"""
import sys
sys.path.insert(0, './src')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold


class ImprovedCNN(nn.Module):
    """Improved CNN with proper architecture and regularization."""
    
    def __init__(self, input_size=360):
        super().__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Calculate flattened size
        self.flat_size = 64 * (input_size // 4)
        
        # Classifier
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = x.flatten(1)
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    metrics = MetricsCalculator.calculate_all(
        np.array(all_targets), np.array(all_preds), 
        np.zeros_like(all_preds)  # Dummy probs for training
    )
    
    return avg_loss, metrics


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    metrics = MetricsCalculator.calculate_all(
        np.array(all_targets), np.array(all_preds), np.array(all_probs)
    )
    
    return avg_loss, metrics


def train_with_config(lr, weight_decay, batch_size, seed, fold_idx=0):
    """Train CNN with given hyperparameters."""
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    config = load_config('./config/config.yaml')
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(data_dir, existing_records, window_size=360, encoder=encoder)
    
    # K-fold split
    kfold = StratifiedPatientKFold(n_splits=5, random_state=42)
    splits = list(kfold.split(dataset))
    train_idx, val_idx = splits[fold_idx]
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2)
    
    # Model
    model = ImprovedCNN(input_size=360).to(device)
    
    # Class weights for imbalance
    targets = dataset.labels[train_idx]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    best_val_f1 = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(100):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['macro_f1'])
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Val F1={val_metrics['macro_f1']:.4f}, Val Sens={val_metrics['sensitivity']:.4f}")
        
        # Early stopping based on macro F1
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 
                      f'./models/improved_cnn_fold{fold_idx}_seed{seed}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(
        f'./models/improved_cnn_fold{fold_idx}_seed{seed}.pth',
        weights_only=True
    ))
    
    _, val_metrics = evaluate(model, val_loader, criterion, device)
    
    return val_metrics, model


def hyperparameter_search():
    """Search for best hyperparameters."""
    
    print("="*70)
    print("CNN Hyperparameter Search")
    print("="*70)
    
    # Search space
    lrs = [1e-4, 5e-4, 1e-3]
    weight_decays = [1e-4, 1e-3]
    batch_sizes = [32, 64]
    
    best_config = None
    best_score = 0
    results = []
    
    for lr in lrs:
        for wd in weight_decays:
            for bs in batch_sizes:
                print(f"\nTesting: lr={lr}, wd={wd}, bs={bs}")
                
                try:
                    metrics, _ = train_with_config(lr, wd, bs, seed=42, fold_idx=0)
                    score = metrics['macro_f1']
                    
                    print(f"  Result: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}, "
                          f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
                    
                    results.append({
                        'lr': lr, 'wd': wd, 'bs': bs,
                        'accuracy': metrics['accuracy'],
                        'macro_f1': metrics['macro_f1'],
                        'sensitivity': metrics['sensitivity'],
                        'specificity': metrics['specificity']
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_config = {'lr': lr, 'wd': wd, 'bs': bs}
                
                except Exception as e:
                    print(f"  Failed: {e}")
    
    print("\n" + "="*70)
    print("Best Configuration")
    print("="*70)
    print(f"LR: {best_config['lr']}")
    print(f"Weight Decay: {best_config['wd']}")
    print(f"Batch Size: {best_config['bs']}")
    print(f"Best Macro F1: {best_score:.4f}")
    
    return best_config, results


if __name__ == "__main__":
    best_config, results = hyperparameter_search()
    
    # Train with best config on multiple seeds
    print("\n" + "="*70)
    print("Training with Best Config on Multiple Seeds")
    print("="*70)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        metrics, _ = train_with_config(
            best_config['lr'], best_config['wd'], best_config['bs'], 
            seed=seed, fold_idx=0
        )
        all_metrics.append(metrics)
        print(f"  Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
    
    # Average results
    print("\n" + "="*70)
    print("Average Results (3 seeds)")
    print("="*70)
    
    for key in ['accuracy', 'macro_f1', 'sensitivity', 'specificity']:
        values = [m[key] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{key}: {mean:.4f} ± {std:.4f}")
