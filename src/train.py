"""
Training script for Temporal 1D-CSNN with proper regularization and evaluation.
"""
import sys
sys.path.insert(0, '/home/ubuntu/ecg_snn_project/src')

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import set_seed, load_config, ExperimentLogger, MetricsCalculator, print_metrics
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold, download_mitdb
from snn_model import TemporalCSNN, BaselineCNN


class FiringRateRegularizer:
    """Regularization to encourage sparse firing."""
    
    def __init__(self, target_rate: float = 0.1, weight: float = 0.001):
        self.target_rate = target_rate
        self.weight = weight
    
    def __call__(self, spike_counts: dict, batch_size: int, num_steps: int):
        """Calculate firing rate regularization loss."""
        total_neurons = 0
        total_spikes = 0
        
        for layer, count in spike_counts.items():
            total_spikes += count
            # Approximate neuron count (simplified)
            total_neurons += count / (self.target_rate + 1e-6)
        
        if total_neurons == 0:
            return torch.tensor(0.0)
        
        actual_rate = total_spikes / (total_neurons * num_steps * batch_size + 1e-6)
        reg_loss = self.weight * (actual_rate - self.target_rate) ** 2
        
        return torch.tensor(reg_loss)


def train_epoch(model, train_loader, optimizer, criterion, device, 
                grad_clip=1.0, firing_reg=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, TemporalCSNN):
            model.enable_spike_tracking()
            spk_out, mem_out = model(data)
            output = mem_out  # Use membrane potential for classification
            
            # Classification loss
            loss = criterion(output, targets)
            
            # Firing rate regularization
            if firing_reg is not None:
                spike_counts = model.get_spike_counts()
                reg_loss = firing_reg(spike_counts, data.size(0), model.num_steps)
                loss = loss + reg_loss
        else:
            output = model(data)
            loss = criterion(output, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    return total_loss / total_samples


def evaluate(model, test_loader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            if isinstance(model, TemporalCSNN):
                _, mem_out = model(data)
                output = mem_out
            else:
                output = model(data)
            
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of abnormal class
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def train_with_kfold(config: dict, model_class, model_name: str):
    """
    Train model with stratified k-fold cross-validation.
    """
    # Setup
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    # Logger
    logger = ExperimentLogger(
        '/home/ubuntu/ecg_snn_project/logs',
        f"{config['experiment']['name']}_{model_name}"
    )
    logger.log_config(config)
    
    # Download data if needed
    data_dir = '/home/ubuntu/ecg_snn_project/data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    
    # Check which records exist
    existing_records = []
    for r in all_records:
        if os.path.exists(os.path.join(data_dir, f"{r}.hea")):
            existing_records.append(r)
    
    if len(existing_records) < 5:
        print("Downloading additional records...")
        download_mitdb(data_dir, all_records[:10])  # Download first 10 for speed
        existing_records = [r for r in all_records[:10] if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    print(f"Using {len(existing_records)} records: {existing_records}")
    
    # Create dataset
    encoder = GradedDeltaEncoder(
        num_levels=config['encoding']['num_levels'],
        threshold_factor=config['encoding']['threshold_factor']
    )
    
    dataset = MITBIHDataset(
        data_dir,
        existing_records,
        window_size=config['data']['window_size'],
        encoder=encoder
    )
    
    # K-fold cross-validation
    kfold = StratifiedPatientKFold(
        n_splits=config['evaluation']['k_folds'],
        shuffle=True,
        random_state=config['experiment']['seed']
    )
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config['evaluation']['k_folds']}")
        print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        print('='*50)
        
        # Create data loaders with class weighting
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        # Weighted sampler for imbalanced data
        train_labels = dataset.labels[train_idx]
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], 
                                  sampler=sampler)
        test_loader = DataLoader(test_subset, batch_size=config['training']['batch_size'])
        
        # Create model
        if model_class == TemporalCSNN:
            model = TemporalCSNN(
                input_size=config['data']['window_size'],
                num_steps=config['model']['num_steps'],
                beta=config['model']['beta'],
                conv1_channels=config['model']['conv1_channels'],
                conv2_channels=config['model']['conv2_channels'],
                kernel_size=config['model']['kernel_size'],
                stride=config['model']['stride'],
                surrogate_slope=config['model']['surrogate_slope']
            ).to(device)
        else:
            model = BaselineCNN(
                input_size=config['data']['window_size'],
                conv1_channels=config['model']['conv1_channels'],
                conv2_channels=config['model']['conv2_channels'],
                kernel_size=config['model']['kernel_size'],
                stride=config['model']['stride']
            ).to(device)
        
        # Loss with class weights
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Firing rate regularizer (only for SNN)
        firing_reg = None
        if model_class == TemporalCSNN:
            firing_reg = FiringRateRegularizer(
                target_rate=0.1,
                weight=config['training']['firing_rate_reg']
            )
        
        # Training loop with early stopping
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(config['training']['epochs']):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_clip=config['training']['grad_clip'],
                firing_reg=firing_reg
            )
            
            # Evaluate
            preds, targets, probs = evaluate(model, test_loader, device)
            metrics = MetricsCalculator.calculate_all(targets, preds, probs)
            
            # Log
            logger.log_epoch(epoch, train_loss, metrics)
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, "
                  f"macro_F1={metrics['macro_f1']:.4f}, "
                  f"PR-AUC={metrics.get('pr_auc', 0):.4f}")
            
            # Learning rate scheduling
            scheduler.step(metrics['macro_f1'])
            
            # Early stopping
            if metrics['macro_f1'] > best_f1:
                best_f1 = metrics['macro_f1']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          f'/home/ubuntu/ecg_snn_project/models/{model_name}_fold{fold}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and final evaluation
        model.load_state_dict(torch.load(
            f'/home/ubuntu/ecg_snn_project/models/{model_name}_fold{fold}_best.pth',
            weights_only=True
        ))
        preds, targets, probs = evaluate(model, test_loader, device)
        final_metrics = MetricsCalculator.calculate_all(targets, preds, probs)
        
        print_metrics(final_metrics, f"Fold {fold+1} Final Results")
        logger.log_evaluation(fold, final_metrics)
        fold_results.append(final_metrics)
    
    # Aggregate results
    aggregated = MetricsCalculator.aggregate_fold_results(fold_results)
    print_metrics(aggregated, "Aggregated K-Fold Results")
    logger.log_final_results(aggregated)
    logger.save()
    
    return aggregated, fold_results


def main():
    # Load config
    config = load_config('/home/ubuntu/ecg_snn_project/config/config.yaml')
    
    print("="*60)
    print("Training Temporal CSNN")
    print("="*60)
    snn_results, snn_folds = train_with_kfold(config, TemporalCSNN, "temporal_csnn")
    
    print("\n" + "="*60)
    print("Training Baseline CNN")
    print("="*60)
    cnn_results, cnn_folds = train_with_kfold(config, BaselineCNN, "baseline_cnn")
    
    # Save comparison
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Metric':<20} {'SNN':<20} {'CNN':<20}")
    print("-"*60)
    for metric in ['accuracy', 'macro_f1', 'pr_auc', 'sensitivity', 'specificity']:
        snn_val = f"{snn_results.get(f'{metric}_mean', 0):.4f} ± {snn_results.get(f'{metric}_std', 0):.4f}"
        cnn_val = f"{cnn_results.get(f'{metric}_mean', 0):.4f} ± {cnn_results.get(f'{metric}_std', 0):.4f}"
        print(f"{metric:<20} {snn_val:<20} {cnn_val:<20}")


if __name__ == "__main__":
    main()
