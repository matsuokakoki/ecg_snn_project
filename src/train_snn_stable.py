"""
Stable SNN training with conservative settings.
"""
import sys
sys.path.insert(0, './src')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN
from train_final_optimized import find_optimal_threshold_aggressive


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        if isinstance(outputs, tuple):
            outputs = outputs[1]
        
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
            
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = MetricsCalculator.calculate_all(
        np.array(all_targets), np.array(all_preds), np.array(all_probs)
    )
    
    return metrics, np.array(all_targets), np.array(all_probs)


def train_snn_stable():
    """Train SNN with stable settings."""
    
    config = load_config('./config/config.yaml')
    set_seed(42)
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(data_dir, existing_records, window_size=360, encoder=encoder)
    
    # K-fold split
    kfold = StratifiedPatientKFold(n_splits=5, random_state=42)
    splits = list(kfold.split(dataset))
    train_idx, val_idx = splits[0]
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # Weighted sampling (moderate oversampling)
    train_targets = dataset.labels[train_idx]
    class_counts = np.bincount(train_targets)
    # Moderate class weights (not too extreme)
    class_weights = np.array([1.0, 3.0])  # 3x weight for abnormal
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_subset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=64, num_workers=0)
    
    # Model
    model = TemporalCSNN(input_size=360).to(device)
    
    # Simple weighted CE loss
    loss_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # Conservative optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training
    best_val_f1 = 0
    patience_counter = 0
    max_patience = 20
    
    print("Training SNN with Stable Settings")
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    print(f"  Class weights: {loss_weights.cpu().numpy()}")
    print()
    
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_targets, val_probs = evaluate_with_threshold(model, val_loader, device, threshold=0.3)
        
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Sens={val_metrics['sensitivity']:.4f}, "
                  f"Spec={val_metrics['specificity']:.4f}, F1={val_metrics['macro_f1']:.4f}, "
                  f"Acc={val_metrics['accuracy']:.4f}")
        
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            torch.save(model.state_dict(), 
                      './models/snn_stable_fold0.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(
        './models/snn_stable_fold0.pth',
        weights_only=True
    ))
    
    # Find optimal threshold
    _, val_targets, val_probs = evaluate_with_threshold(model, val_loader, device, threshold=0.5)
    optimal_threshold = find_optimal_threshold_aggressive(val_targets, val_probs, target_sensitivity=0.80)
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    # Final evaluation
    final_metrics, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=optimal_threshold)
    
    print(f"\nFinal Results:")
    print(f"  Sensitivity: {final_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {final_metrics['specificity']:.4f}")
    print(f"  Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  PR-AUC: {final_metrics['pr_auc']:.4f}")
    
    passing = (final_metrics['sensitivity'] >= 0.80 and 
               final_metrics['specificity'] >= 0.90 and 
               final_metrics['macro_f1'] >= 0.75)
    
    print(f"  Status: {'✓ PASS' if passing else '✗ FAIL'}")
    
    return final_metrics, optimal_threshold


if __name__ == "__main__":
    print("="*70)
    print("Stable SNN Training")
    print("="*70)
    print()
    
    metrics, threshold = train_snn_stable()
