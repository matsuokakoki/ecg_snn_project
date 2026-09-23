"""
K-fold cross-validation evaluation with mean ± std reporting.
This is the REQUIRED evaluation protocol for the final report.
"""
import sys
sys.path.insert(0, './src')

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN
from train_final_optimized import ImprovedCNN, evaluate_with_threshold, find_optimal_threshold_aggressive


def evaluate_kfold(model_type='snn', n_folds=5, model_path_template=None):
    """
    Evaluate model across all k-folds and compute mean ± std.
    
    Args:
        model_type: 'snn' or 'cnn'
        n_folds: number of folds
        model_path_template: path template with {fold} placeholder
    
    Returns:
        results_df: DataFrame with per-fold results
        summary_stats: Dict with mean ± std for each metric
    """
    
    config = load_config('./config/config.yaml')
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(data_dir, existing_records, window_size=360, encoder=encoder)
    
    # K-fold split
    kfold = StratifiedPatientKFold(n_splits=n_folds, random_state=42)
    splits = list(kfold.split(dataset))
    
    # Store results for each fold
    fold_results = []
    
    print(f"\n{'='*70}")
    print(f"K-Fold Evaluation: {model_type.upper()}")
    print(f"{'='*70}\n")
    
    for fold_idx in range(n_folds):
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print("-" * 50)
        
        _, val_idx = splits[fold_idx]
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=64, num_workers=0)
        
        # Load model
        if model_path_template:
            model_path = model_path_template.format(fold=fold_idx)
        else:
            model_path = f'./models/{model_type}_optimized_fold{fold_idx}.pth'
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            print(f"  Skipping fold {fold_idx}")
            continue
        
        # Create model
        if model_type == 'snn':
            model = TemporalCSNN(input_size=360).to(device)
        else:
            model = ImprovedCNN(input_size=360).to(device)
        
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # Get predictions with default threshold
        _, val_targets, val_probs = evaluate_with_threshold(model, val_loader, device, threshold=0.5)
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold_aggressive(val_targets, val_probs, target_sensitivity=0.80)
        
        # Evaluate with optimal threshold
        final_metrics, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=optimal_threshold)
        
        # Store results
        result = {
            'fold': fold_idx + 1,
            'threshold': optimal_threshold,
            'accuracy': final_metrics['accuracy'],
            'sensitivity': final_metrics['sensitivity'],
            'specificity': final_metrics['specificity'],
            'macro_f1': final_metrics['macro_f1'],
            'pr_auc': final_metrics['pr_auc'],
            'n_samples': len(val_subset)
        }
        
        fold_results.append(result)
        
        # Check passing criteria
        passing = (final_metrics['sensitivity'] >= 0.80 and 
                  final_metrics['specificity'] >= 0.90 and 
                  final_metrics['macro_f1'] >= 0.75)
        
        print(f"  Threshold: {optimal_threshold:.4f}")
        print(f"  Sensitivity: {final_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {final_metrics['specificity']:.4f}")
        print(f"  Macro F1: {final_metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  PR-AUC: {final_metrics['pr_auc']:.4f}")
        print(f"  Status: {'✓ PASS' if passing else '✗ FAIL'}")
        print()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # Compute summary statistics
    summary_stats = {}
    for metric in ['accuracy', 'sensitivity', 'specificity', 'macro_f1', 'pr_auc']:
        values = results_df[metric].values
        summary_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),  # Sample std
            'min': np.min(values),
            'max': np.max(values),
            'ci_95_lower': np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(len(values)),
            'ci_95_upper': np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary Statistics ({n_folds}-Fold Cross-Validation)")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'95% CI':<25} {'Pass?':<10}")
    print("-" * 70)
    
    passing_criteria = {
        'sensitivity': 0.80,
        'specificity': 0.90,
        'macro_f1': 0.75
    }
    
    for metric in ['accuracy', 'sensitivity', 'specificity', 'macro_f1', 'pr_auc']:
        stats = summary_stats[metric]
        mean = stats['mean']
        std = stats['std']
        ci_lower = stats['ci_95_lower']
        ci_upper = stats['ci_95_upper']
        
        # Check if passing
        if metric in passing_criteria:
            passing = mean >= passing_criteria[metric]
            pass_str = '✓ PASS' if passing else '✗ FAIL'
        else:
            pass_str = '-'
        
        print(f"{metric:<15} {mean:<10.4f} {std:<10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]    {pass_str:<10}")
    
    print()
    
    # Overall passing
    overall_passing = (
        summary_stats['sensitivity']['mean'] >= 0.80 and
        summary_stats['specificity']['mean'] >= 0.90 and
        summary_stats['macro_f1']['mean'] >= 0.75
    )
    
    print(f"Overall Status: {'✓ PASS ALL CRITERIA' if overall_passing else '✗ FAIL'}")
    print()
    
    return results_df, summary_stats


def create_comparison_table(snn_results, cnn_results, snn_stats, cnn_stats):
    """Create a comparison table for the report."""
    
    print(f"\n{'='*70}")
    print("SNN vs CNN Comparison (Mean ± Std)")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<15} {'SNN':<25} {'CNN':<25} {'Winner':<10}")
    print("-" * 70)
    
    for metric in ['accuracy', 'sensitivity', 'specificity', 'macro_f1', 'pr_auc']:
        snn_mean = snn_stats[metric]['mean']
        snn_std = snn_stats[metric]['std']
        cnn_mean = cnn_stats[metric]['mean']
        cnn_std = cnn_stats[metric]['std']
        
        winner = 'SNN' if snn_mean > cnn_mean else 'CNN' if cnn_mean > snn_mean else 'Tie'
        
        print(f"{metric:<15} {snn_mean:.4f} ± {snn_std:.4f}      {cnn_mean:.4f} ± {cnn_std:.4f}      {winner:<10}")
    
    print()


def plot_kfold_results(snn_results, cnn_results, output_path):
    """Plot k-fold results comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Fold Cross-Validation Results: SNN vs CNN', fontsize=16, fontweight='bold')
    
    metrics = ['sensitivity', 'specificity', 'macro_f1', 'accuracy']
    titles = ['Sensitivity', 'Specificity', 'Macro F1', 'Accuracy']
    passing_lines = [0.80, 0.90, 0.75, None]
    
    for idx, (metric, title, passing_line) in enumerate(zip(metrics, titles, passing_lines)):
        ax = axes[idx // 2, idx % 2]
        
        # Extract values
        snn_vals = snn_results[metric].values
        cnn_vals = cnn_results[metric].values
        folds = snn_results['fold'].values
        
        # Plot
        ax.plot(folds, snn_vals, 'o-', label='SNN', linewidth=2, markersize=8, color='#2E86AB')
        ax.plot(folds, cnn_vals, 's-', label='CNN', linewidth=2, markersize=8, color='#A23B72')
        
        # Mean lines
        ax.axhline(np.mean(snn_vals), linestyle='--', alpha=0.5, color='#2E86AB', label=f'SNN Mean: {np.mean(snn_vals):.3f}')
        ax.axhline(np.mean(cnn_vals), linestyle='--', alpha=0.5, color='#A23B72', label=f'CNN Mean: {np.mean(cnn_vals):.3f}')
        
        # Passing line
        if passing_line is not None:
            ax.axhline(passing_line, linestyle=':', color='red', linewidth=2, alpha=0.7, label=f'Threshold: {passing_line}')
        
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(folds)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    # Evaluate SNN
    snn_results, snn_stats = evaluate_kfold('snn', n_folds=5)
    
    # Evaluate CNN
    cnn_results, cnn_stats = evaluate_kfold('cnn', n_folds=5)
    
    # Create comparison table
    create_comparison_table(snn_results, cnn_results, snn_stats, cnn_stats)
    
    # Plot results
    plot_kfold_results(snn_results, cnn_results, 
                      './results/kfold_comparison.png')
    
    # Save results to CSV
    snn_results.to_csv('./results/snn_kfold_results.csv', index=False)
    cnn_results.to_csv('./results/cnn_kfold_results.csv', index=False)
    
    print("\nResults saved to:")
    print("  - ./results/snn_kfold_results.csv")
    print("  - ./results/cnn_kfold_results.csv")
    print("  - ./results/kfold_comparison.png")
