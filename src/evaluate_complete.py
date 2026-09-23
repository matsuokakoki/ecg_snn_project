"""
Complete evaluation script with SOPs measurement and CNN comparison.
"""
import sys
sys.path.insert(0, './src')

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import set_seed, load_config, MetricsCalculator, print_metrics
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN, BaselineCNN


def measure_sops(model, data_loader, device):
    """Measure SOPs for SNN model."""
    model.eval()
    model.enable_spike_tracking()
    
    total_sops = {'layer1': 0, 'layer2': 0, 'fc': 0, 'total': 0}
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _, _ = model(data)
            
            sops = model.calculate_sops()
            for key in total_sops:
                total_sops[key] += sops.get(key, 0)
            total_samples += data.size(0)
    
    # Average per sample
    avg_sops = {k: v / total_samples for k, v in total_sops.items()}
    
    return avg_sops, total_sops


def evaluate_models():
    """Complete evaluation of SNN and CNN models."""
    config = load_config('./config/config.yaml')
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=config['encoding']['num_levels'], threshold_factor=config['encoding']['threshold_factor'])
    dataset = MITBIHDataset(data_dir, existing_records, window_size=config['data']['window_size'], encoder=encoder)
    
    # Use a simple train/test split for quick evaluation
    kfold = StratifiedPatientKFold(n_splits=5, random_state=config['experiment']['seed'])
    train_idx, test_idx = next(kfold.split(dataset))
    
    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=64)
    
    # Create models
    snn_model = TemporalCSNN(input_size=config['data']['window_size']).to(device)
    cnn_model = BaselineCNN(input_size=config['data']['window_size']).to(device)
    
    # Load best models if available
    snn_path = './models/temporal_csnn_v2_fold0_best.pth'
    cnn_path = './models/baseline_cnn_v2_fold0_best.pth'
    
    if os.path.exists(snn_path):
        snn_model.load_state_dict(torch.load(snn_path, weights_only=True))
        print("Loaded SNN model")
    
    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, weights_only=True))
        print("Loaded CNN model")
    
    # Evaluate SNN
    print("\n" + "="*60)
    print("SNN Evaluation")
    print("="*60)
    
    snn_model.eval()
    snn_preds, snn_targets, snn_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            _, output = snn_model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            snn_preds.extend(preds.cpu().numpy())
            snn_targets.extend(targets.numpy())
            snn_probs.extend(probs[:, 1].cpu().numpy())
    
    snn_metrics = MetricsCalculator.calculate_all(np.array(snn_targets), np.array(snn_preds), np.array(snn_probs))
    print_metrics(snn_metrics, "SNN Classification Metrics")
    
    # Measure SOPs
    avg_sops, total_sops = measure_sops(snn_model, test_loader, device)
    print(f"\nSNN SOPs (per sample):")
    for layer, sops in avg_sops.items():
        print(f"  {layer}: {sops:.2f}")
    
    # Evaluate CNN
    print("\n" + "="*60)
    print("CNN Evaluation")
    print("="*60)
    
    cnn_model.eval()
    cnn_preds, cnn_targets, cnn_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            output = cnn_model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            cnn_preds.extend(preds.cpu().numpy())
            cnn_targets.extend(targets.numpy())
            cnn_probs.extend(probs[:, 1].cpu().numpy())
    
    cnn_metrics = MetricsCalculator.calculate_all(np.array(cnn_targets), np.array(cnn_preds), np.array(cnn_probs))
    print_metrics(cnn_metrics, "CNN Classification Metrics")
    
    cnn_flops = cnn_model.count_flops()
    print(f"\nCNN FLOPs (per sample): {cnn_flops}")
    
    # Comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'SNN':<15} {'CNN':<15}")
    print("-"*50)
    print(f"{'Parameters':<20} {snn_model.count_parameters():<15} {cnn_model.count_parameters():<15}")
    print(f"{'Accuracy':<20} {snn_metrics['accuracy']:.4f}{'':>9} {cnn_metrics['accuracy']:.4f}")
    print(f"{'Macro F1':<20} {snn_metrics['macro_f1']:.4f}{'':>9} {cnn_metrics['macro_f1']:.4f}")
    print(f"{'Sensitivity':<20} {snn_metrics.get('sensitivity', 0):.4f}{'':>9} {cnn_metrics.get('sensitivity', 0):.4f}")
    print(f"{'Specificity':<20} {snn_metrics.get('specificity', 0):.4f}{'':>9} {cnn_metrics.get('specificity', 0):.4f}")
    print(f"{'PR-AUC':<20} {snn_metrics.get('pr_auc', 0):.4f}{'':>9} {cnn_metrics.get('pr_auc', 0):.4f}")
    print(f"{'Operations/sample':<20} {avg_sops['total']:.0f} SOPs{'':>4} {cnn_flops} FLOPs")
    
    efficiency_ratio = cnn_flops / max(avg_sops['total'], 1)
    print(f"\nEfficiency Ratio (FLOPs/SOPs): {efficiency_ratio:.1f}x")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Metrics comparison
    metrics_names = ['Accuracy', 'Macro F1', 'Sensitivity', 'Specificity']
    snn_values = [snn_metrics['accuracy'], snn_metrics['macro_f1'], 
                  snn_metrics.get('sensitivity', 0), snn_metrics.get('specificity', 0)]
    cnn_values = [cnn_metrics['accuracy'], cnn_metrics['macro_f1'],
                  cnn_metrics.get('sensitivity', 0), cnn_metrics.get('specificity', 0)]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0].bar(x - width/2, snn_values, width, label='SNN', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, cnn_values, width, label='CNN', color='orange', alpha=0.7)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classification Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, rotation=45)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    
    # Operations comparison
    axes[1].bar(['SNN (SOPs)', 'CNN (FLOPs)'], [avg_sops['total'], cnn_flops], 
                color=['blue', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Operations per sample')
    axes[1].set_title('Computational Cost')
    axes[1].set_yscale('log')
    
    # SOPs breakdown
    sops_layers = ['Layer 1', 'Layer 2', 'FC']
    sops_values = [avg_sops['layer1'], avg_sops['layer2'], avg_sops['fc']]
    axes[2].bar(sops_layers, sops_values, color='blue', alpha=0.7)
    axes[2].set_ylabel('SOPs per sample')
    axes[2].set_title('SNN SOPs by Layer')
    
    plt.tight_layout()
    plt.savefig('./results/complete_comparison.png', dpi=150)
    print("\nSaved comparison plot to results/complete_comparison.png")
    
    return snn_metrics, cnn_metrics, avg_sops, cnn_flops


if __name__ == "__main__":
    evaluate_models()
