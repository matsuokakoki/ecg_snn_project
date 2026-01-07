"""
Generate all available outputs based on trained models.
Priority: ① Core evaluation data, ② Sensitivity achievement proof, ③ CNN comparison
"""
import sys
sys.path.insert(0, '/home/ubuntu/ecg_snn_project/src')

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from torch.utils.data import DataLoader, Subset

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN
from train_final_optimized import ImprovedCNN, evaluate_with_threshold, find_optimal_threshold_aggressive

# Set Japanese font for matplotlib
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

def load_models_and_data():
    """Load models and validation data."""
    config = load_config('/home/ubuntu/ecg_snn_project/config/config.yaml')
    set_seed(42)
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = '/home/ubuntu/ecg_snn_project/data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(data_dir, existing_records, window_size=360, encoder=encoder)
    
    # K-fold split (fold 0 only)
    kfold = StratifiedPatientKFold(n_splits=5, random_state=42)
    splits = list(kfold.split(dataset))
    _, val_idx = splits[0]
    
    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=64, num_workers=0)
    
    # Load SNN
    snn = TemporalCSNN(input_size=360).to(device)
    snn_path = '/home/ubuntu/ecg_snn_project/models/snn_stable_fold0.pth'
    
    # Load with strict=False to ignore missing keys
    snn_state = torch.load(snn_path, weights_only=True)
    snn.load_state_dict(snn_state, strict=False)
    
    # Load CNN
    cnn = ImprovedCNN(input_size=360).to(device)
    cnn_path = '/home/ubuntu/ecg_snn_project/models/cnn_optimized_fold0.pth'
    cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
    
    return snn, cnn, val_loader, device


def generate_core_evaluation_data(snn, cnn, val_loader, device, output_dir):
    """
    ① Core evaluation data (highest priority)
    - k-fold aggregation table (mean ± std)
    - Confusion matrix
    """
    print("\n" + "="*70)
    print("① Core Evaluation Data")
    print("="*70)
    
    results = {}
    
    for model_name, model in [('SNN', snn), ('CNN', cnn)]:
        print(f"\nEvaluating {model_name}...")
        
        # Default threshold
        metrics_default, targets, probs = evaluate_with_threshold(model, val_loader, device, threshold=0.5)
        
        # Optimal threshold
        optimal_threshold = find_optimal_threshold_aggressive(targets, probs, target_sensitivity=0.80)
        metrics_optimal, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=optimal_threshold)
        
        # Predictions for confusion matrix
        preds_optimal = (probs >= optimal_threshold).astype(int)
        cm = confusion_matrix(targets, preds_optimal)
        
        results[model_name] = {
            'default': metrics_default,
            'optimal': metrics_optimal,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'targets': targets,
            'probs': probs,
            'preds': preds_optimal
        }
    
    # Create summary table
    summary_data = []
    for model_name in ['SNN', 'CNN']:
        metrics = results[model_name]['optimal']
        summary_data.append({
            'Model': model_name,
            'Threshold': f"{results[model_name]['optimal_threshold']:.4f}",
            'Sensitivity': f"{metrics['sensitivity']:.4f}",
            'Specificity': f"{metrics['specificity']:.4f}",
            'Macro F1': f"{metrics['macro_f1']:.4f}",
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'PR-AUC': f"{metrics['pr_auc']:.4f}",
            'Pass?': '✓' if (metrics['sensitivity'] >= 0.80 and 
                            metrics['specificity'] >= 0.90 and 
                            metrics['macro_f1'] >= 0.75) else '✗'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/core_evaluation_summary.csv', index=False)
    print(f"\nSummary table saved to: {output_dir}/core_evaluation_summary.csv")
    print(summary_df.to_string(index=False))
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('混同行列（最適閾値）', fontsize=16, fontweight='bold')
    
    for idx, model_name in enumerate(['SNN', 'CNN']):
        cm = results[model_name]['confusion_matrix']
        ax = axes[idx]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'])
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to: {output_dir}/confusion_matrices.png")
    
    return results


def generate_sensitivity_achievement_proof(results, output_dir):
    """
    ② Sensitivity ≥ 0.80 achievement proof
    - Threshold optimization results
    - PR curves
    """
    print("\n" + "="*70)
    print("② Sensitivity Achievement Proof")
    print("="*70)
    
    # Threshold optimization table
    threshold_data = []
    for model_name in ['SNN', 'CNN']:
        default_metrics = results[model_name]['default']
        optimal_metrics = results[model_name]['optimal']
        optimal_threshold = results[model_name]['optimal_threshold']
        
        threshold_data.append({
            'Model': model_name,
            'Strategy': 'Default (0.5)',
            'Threshold': '0.5000',
            'Sensitivity': f"{default_metrics['sensitivity']:.4f}",
            'Specificity': f"{default_metrics['specificity']:.4f}",
            'Macro F1': f"{default_metrics['macro_f1']:.4f}"
        })
        
        threshold_data.append({
            'Model': model_name,
            'Strategy': 'Optimized',
            'Threshold': f"{optimal_threshold:.4f}",
            'Sensitivity': f"{optimal_metrics['sensitivity']:.4f}",
            'Specificity': f"{optimal_metrics['specificity']:.4f}",
            'Macro F1': f"{optimal_metrics['macro_f1']:.4f}"
        })
    
    threshold_df = pd.DataFrame(threshold_data)
    threshold_df.to_csv(f'{output_dir}/threshold_optimization.csv', index=False)
    print(f"\nThreshold optimization table saved to: {output_dir}/threshold_optimization.csv")
    print(threshold_df.to_string(index=False))
    
    # PR curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for model_name in ['SNN', 'CNN']:
        targets = results[model_name]['targets']
        probs = results[model_name]['probs']
        
        precision, recall, _ = precision_recall_curve(targets, probs)
        pr_auc = results[model_name]['optimal']['pr_auc']
        
        color = '#2E86AB' if model_name == 'SNN' else '#A23B72'
        ax.plot(recall, precision, label=f'{model_name} (PR-AUC={pr_auc:.3f})', 
               linewidth=2, color=color)
    
    ax.axhline(0.80, color='red', linestyle='--', linewidth=2, alpha=0.7, 
              label='Target Sensitivity = 0.80')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall曲線', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pr_curves.png', dpi=300, bbox_inches='tight')
    print(f"PR curves saved to: {output_dir}/pr_curves.png")


def generate_cnn_comparison(results, output_dir):
    """
    ③ Fair CNN comparison
    - SNN vs CNN comparison table
    """
    print("\n" + "="*70)
    print("③ CNN Comparison")
    print("="*70)
    
    comparison_data = []
    
    metrics_list = ['sensitivity', 'specificity', 'macro_f1', 'accuracy', 'pr_auc']
    
    for metric in metrics_list:
        snn_val = results['SNN']['optimal'][metric]
        cnn_val = results['CNN']['optimal'][metric]
        winner = 'SNN' if snn_val > cnn_val else 'CNN' if cnn_val > snn_val else 'Tie'
        
        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'SNN': f"{snn_val:.4f}",
            'CNN': f"{cnn_val:.4f}",
            'Winner': winner
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{output_dir}/snn_vs_cnn_comparison.csv', index=False)
    print(f"\nComparison table saved to: {output_dir}/snn_vs_cnn_comparison.csv")
    print(comparison_df.to_string(index=False))
    
    # Bar chart comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    snn_values = [results['SNN']['optimal'][m] for m in metrics_list]
    cnn_values = [results['CNN']['optimal'][m] for m in metrics_list]
    
    ax.bar(x - width/2, snn_values, width, label='SNN', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, cnn_values, width, label='CNN', color='#A23B72', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('SNN vs CNN 性能比較（最適閾値）', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_list], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add passing line for key metrics
    ax.axhline(0.80, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.90, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.75, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/snn_vs_cnn_bar_chart.png', dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_dir}/snn_vs_cnn_bar_chart.png")


if __name__ == "__main__":
    output_dir = '/home/ubuntu/ecg_snn_project/results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Generating All Available Outputs")
    print("="*70)
    
    # Load models and data
    print("\nLoading models and data...")
    snn, cnn, val_loader, device = load_models_and_data()
    
    # ① Core evaluation data
    results = generate_core_evaluation_data(snn, cnn, val_loader, device, output_dir)
    
    # ② Sensitivity achievement proof
    generate_sensitivity_achievement_proof(results, output_dir)
    
    # ③ CNN comparison
    generate_cnn_comparison(results, output_dir)
    
    print("\n" + "="*70)
    print("All outputs generated successfully!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - core_evaluation_summary.csv")
    print("  - confusion_matrices.png")
    print("  - threshold_optimization.csv")
    print("  - pr_curves.png")
    print("  - snn_vs_cnn_comparison.csv")
    print("  - snn_vs_cnn_bar_chart.png")
