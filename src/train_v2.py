"""
Improved training script for ECG classification with SNN and CNN.
"""
import sys
sys.path.insert(0, './src')

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import set_seed, load_config, ExperimentLogger, MetricsCalculator, print_metrics
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold, download_mitdb
from snn_model_v2 import TemporalCSNN, BaselineCNN


def train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=1.0, is_snn=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if is_snn:
            _, output = model(data)
        else:
            output = model(data)
        
        loss = criterion(output, targets)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    return total_loss / total_samples


def evaluate(model, test_loader, device, is_snn=False):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            if is_snn:
                _, output = model(data)
            else:
                output = model(data)
            
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def train_model(model_class, model_name, dataset, config, is_snn=False):
    """Train a single model with k-fold cross-validation."""
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    logger = ExperimentLogger(
        './logs',
        f"{config['experiment']['name']}_{model_name}"
    )
    logger.log_config(config)
    
    kfold = StratifiedPatientKFold(n_splits=config['evaluation']['k_folds'], random_state=config['experiment']['seed'])
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config['evaluation']['k_folds']}")
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
        print('='*50)
        
        # Data loaders
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_labels = dataset.labels[train_idx]
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], sampler=sampler)
        test_loader = DataLoader(test_subset, batch_size=config['training']['batch_size'])
        
        # Create model
        model = model_class(input_size=config['data']['window_size']).to(device)
        
        # Loss and optimizer
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(config['training']['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                    grad_clip=config['training']['grad_clip'], is_snn=is_snn)
            
            preds, targets, probs = evaluate(model, test_loader, device, is_snn=is_snn)
            metrics = MetricsCalculator.calculate_all(targets, preds, probs)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['macro_f1']:.4f}, Sens={metrics.get('sensitivity', 0):.4f}")
            
            if metrics['macro_f1'] > best_f1:
                best_f1 = metrics['macro_f1']
                patience_counter = 0
                torch.save(model.state_dict(), f'./models/{model_name}_fold{fold}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final evaluation
        model.load_state_dict(torch.load(f'./models/{model_name}_fold{fold}_best.pth', weights_only=True))
        preds, targets, probs = evaluate(model, test_loader, device, is_snn=is_snn)
        final_metrics = MetricsCalculator.calculate_all(targets, preds, probs)
        
        print_metrics(final_metrics, f"Fold {fold+1} Final")
        logger.log_evaluation(fold, final_metrics)
        fold_results.append(final_metrics)
    
    aggregated = MetricsCalculator.aggregate_fold_results(fold_results)
    logger.log_final_results(aggregated)
    logger.save()
    
    return aggregated, fold_results


def main():
    config = load_config('./config/config.yaml')
    
    # Prepare data
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    if len(existing_records) < 4:
        print("Downloading data...")
        download_mitdb(data_dir, all_records[:8])
        existing_records = [r for r in all_records[:8] if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    print(f"Using {len(existing_records)} records")
    
    encoder = GradedDeltaEncoder(num_levels=config['encoding']['num_levels'], threshold_factor=config['encoding']['threshold_factor'])
    dataset = MITBIHDataset(data_dir, existing_records, window_size=config['data']['window_size'], encoder=encoder)
    
    # Train SNN
    print("\n" + "="*60)
    print("Training Temporal CSNN")
    print("="*60)
    snn_results, _ = train_model(TemporalCSNN, "temporal_csnn_v2", dataset, config, is_snn=True)
    
    # Train CNN
    print("\n" + "="*60)
    print("Training Baseline CNN")
    print("="*60)
    cnn_results, _ = train_model(BaselineCNN, "baseline_cnn_v2", dataset, config, is_snn=False)
    
    # Comparison
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Metric':<20} {'SNN':<25} {'CNN':<25}")
    print("-"*70)
    for metric in ['accuracy', 'macro_f1', 'pr_auc', 'sensitivity', 'specificity']:
        snn_val = f"{snn_results.get(f'{metric}_mean', 0):.4f} ± {snn_results.get(f'{metric}_std', 0):.4f}"
        cnn_val = f"{cnn_results.get(f'{metric}_mean', 0):.4f} ± {cnn_results.get(f'{metric}_std', 0):.4f}"
        print(f"{metric:<20} {snn_val:<25} {cnn_val:<25}")
    
    # Save comparison plot
    metrics_to_plot = ['accuracy', 'macro_f1', 'sensitivity', 'specificity']
    snn_means = [snn_results.get(f'{m}_mean', 0) for m in metrics_to_plot]
    cnn_means = [cnn_results.get(f'{m}_mean', 0) for m in metrics_to_plot]
    snn_stds = [snn_results.get(f'{m}_std', 0) for m in metrics_to_plot]
    cnn_stds = [cnn_results.get(f'{m}_std', 0) for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, snn_means, width, yerr=snn_stds, label='SNN', capsize=5)
    ax.bar(x + width/2, cnn_means, width, yerr=cnn_stds, label='CNN', capsize=5)
    ax.set_ylabel('Score')
    ax.set_title('SNN vs CNN Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('./results/model_comparison.png', dpi=150)
    print("\nSaved comparison plot to results/model_comparison.png")


if __name__ == "__main__":
    main()
