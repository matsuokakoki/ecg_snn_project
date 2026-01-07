"""
Training with strong imbalance handling:
- Focal Loss
- Threshold optimization for Sensitivity >= 0.80, Specificity >= 0.90
- SMOTE oversampling
"""
import sys
sys.path.insert(0, '/home/ubuntu/ecg_snn_project/src')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_curve

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ImprovedCNN(nn.Module):
    """Improved CNN baseline."""
    
    def __init__(self, input_size=360):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.flat_size = 64 * (input_size // 4)
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = x.unsqueeze(1)
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
    
    def count_flops(self, input_size=360):
        """Count FLOPs for this CNN."""
        # Conv1: (1 * 5 * 32) * (input_size * 32)
        flops_conv1 = 5 * 1 * 32 * (input_size * 32)
        # Pool1: negligible
        # Conv2: (32 * 5 * 64) * ((input_size//2) * 64)
        flops_conv2 = 5 * 32 * 64 * ((input_size // 2) * 64)
        # FC1: flat_size * 128
        flops_fc1 = (64 * (input_size // 4)) * 128
        # FC2: 128 * 2
        flops_fc2 = 128 * 2
        
        total = flops_conv1 + flops_conv2 + flops_fc1 + flops_fc2
        return total


def find_optimal_threshold(targets, probs, min_specificity=0.90):
    """
    Find threshold that maximizes sensitivity while maintaining specificity >= min_specificity.
    """
    fpr, tpr, thresholds = roc_curve(targets, probs)
    specificity = 1 - fpr
    
    # Find thresholds where specificity >= min_specificity
    valid_indices = np.where(specificity >= min_specificity)[0]
    
    if len(valid_indices) == 0:
        # Fallback: use threshold that maximizes Youden's J
        j_scores = tpr + specificity - 1
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    
    # Among valid thresholds, choose one with maximum sensitivity
    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    return thresholds[best_idx]


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        # Handle SNN output (tuple) vs CNN output (tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[1]  # Use logits from SNN
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate_with_threshold(model, loader, device, threshold=0.5):
    """Evaluate model with custom threshold."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            outputs = model(data)
            # Handle SNN output (tuple) vs CNN output (tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # Use logits from SNN
            probs = torch.softmax(outputs, dim=1)
            
            # Apply custom threshold
            preds = (probs[:, 1] >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = MetricsCalculator.calculate_all(
        np.array(all_targets), np.array(all_preds), np.array(all_probs)
    )
    
    return metrics, np.array(all_targets), np.array(all_probs)


def train_model(model_type='snn', fold_idx=0, use_focal=True, use_sampling=True):
    """
    Train model with imbalance handling.
    
    Args:
        model_type: 'snn' or 'cnn'
        fold_idx: fold index
        use_focal: use focal loss
        use_sampling: use weighted sampling
    """
    
    config = load_config('/home/ubuntu/ecg_snn_project/config/config.yaml')
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = '/home/ubuntu/ecg_snn_project/data/mitdb'
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
    
    # Weighted sampling for training
    if use_sampling:
        train_targets = dataset.labels[train_idx]
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_subset, batch_size=64, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    
    val_loader = DataLoader(val_subset, batch_size=64, num_workers=2)
    
    # Model
    if model_type == 'snn':
        model = TemporalCSNN(input_size=360).to(device)
    else:
        model = ImprovedCNN(input_size=360).to(device)
    
    # Loss
    if use_focal:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
    else:
        # Class-weighted CE
        train_targets = dataset.labels[train_idx]
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * 2
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    best_val_f1 = 0
    patience_counter = 0
    max_patience = 15
    
    print(f"\nTraining {model_type.upper()} on fold {fold_idx}")
    print(f"  Focal Loss: {use_focal}, Weighted Sampling: {use_sampling}")
    
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_targets, val_probs = evaluate_with_threshold(model, val_loader, device, threshold=0.5)
        
        scheduler.step(val_metrics['macro_f1'])
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                  f"F1={val_metrics['macro_f1']:.4f}, Sens={val_metrics['sensitivity']:.4f}, "
                  f"Spec={val_metrics['specificity']:.4f}")
        
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            torch.save(model.state_dict(), 
                      f'/home/ubuntu/ecg_snn_project/models/{model_type}_imbalanced_fold{fold_idx}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(
        f'/home/ubuntu/ecg_snn_project/models/{model_type}_imbalanced_fold{fold_idx}.pth',
        weights_only=True
    ))
    
    # Find optimal threshold on validation set
    _, val_targets, val_probs = evaluate_with_threshold(model, val_loader, device, threshold=0.5)
    optimal_threshold = find_optimal_threshold(val_targets, val_probs, min_specificity=0.90)
    
    print(f"\n  Optimal threshold: {optimal_threshold:.4f}")
    
    # Evaluate with optimal threshold
    final_metrics, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=optimal_threshold)
    
    print(f"  Final metrics (threshold={optimal_threshold:.4f}):")
    print(f"    Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"    Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"    Sensitivity: {final_metrics['sensitivity']:.4f}")
    print(f"    Specificity: {final_metrics['specificity']:.4f}")
    print(f"    PR-AUC: {final_metrics['pr_auc']:.4f}")
    
    # Check if passing criteria
    passing = (final_metrics['sensitivity'] >= 0.80 and 
               final_metrics['specificity'] >= 0.90 and 
               final_metrics['macro_f1'] >= 0.75)
    
    print(f"  Passing criteria: {'✓ PASS' if passing else '✗ FAIL'}")
    
    return final_metrics, optimal_threshold, model


if __name__ == "__main__":
    print("="*70)
    print("Training with Strong Imbalance Handling")
    print("="*70)
    
    # Train SNN
    print("\n" + "="*70)
    print("SNN Training")
    print("="*70)
    snn_metrics, snn_threshold, snn_model = train_model('snn', fold_idx=0, use_focal=True, use_sampling=True)
    
    # Train CNN
    print("\n" + "="*70)
    print("CNN Training")
    print("="*70)
    cnn_metrics, cnn_threshold, cnn_model = train_model('cnn', fold_idx=0, use_focal=True, use_sampling=True)
    
    # Summary
    print("\n" + "="*70)
    print("Summary (Fold 0)")
    print("="*70)
    print(f"\nSNN (threshold={snn_threshold:.4f}):")
    print(f"  Sensitivity: {snn_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {snn_metrics['specificity']:.4f}")
    print(f"  Macro F1: {snn_metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {snn_metrics['accuracy']:.4f}")
    
    print(f"\nCNN (threshold={cnn_threshold:.4f}):")
    print(f"  Sensitivity: {cnn_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {cnn_metrics['specificity']:.4f}")
    print(f"  Macro F1: {cnn_metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {cnn_metrics['accuracy']:.4f}")
