"""
Detailed SOPs analysis with layer-by-layer breakdown and comparison with CNN FLOPs.
Goal: Demonstrate SOPs/FLOPs <= 1 for power efficiency claim.
"""
import sys
sys.path.insert(0, '/home/ubuntu/ecg_snn_project/src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from utils import set_seed, load_config
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN
from train_final_optimized import ImprovedCNN


def count_snn_sops_detailed(model, data_loader, device):
    """
    Count SOPs for SNN with layer-by-layer breakdown.
    
    SOPs definition:
    - Conv: output_spikes × (kernel_size × in_channels)
    - FC: output_spikes × in_features
    """
    
    model.eval()
    
    layer_sops = {
        'conv1': [],
        'conv2': [],
        'conv3': [],
        'fc1': [],
        'fc2': []
    }
    
    layer_spike_rates = {
        'conv1': [],
        'conv2': [],
        'conv3': [],
        'fc1': [],
        'fc2': []
    }
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            batch_size = data.size(0)
            
            # Forward pass with spike tracking
            spikes_dict, _ = model(data)
            
            # Conv1: (batch, 32, T, L)
            conv1_spikes = spikes_dict['conv1']  # (batch, 32, T, L)
            n_spikes_conv1 = conv1_spikes.sum().item()
            spike_rate_conv1 = conv1_spikes.float().mean().item()
            
            # SOPs for Conv1: n_spikes × (kernel_size × in_channels)
            # Conv1: in=1, kernel=5
            sops_conv1 = n_spikes_conv1 * (5 * 1)
            
            layer_sops['conv1'].append(sops_conv1 / batch_size)
            layer_spike_rates['conv1'].append(spike_rate_conv1)
            
            # Conv2
            conv2_spikes = spikes_dict['conv2']
            n_spikes_conv2 = conv2_spikes.sum().item()
            spike_rate_conv2 = conv2_spikes.float().mean().item()
            # Conv2: in=32, kernel=5
            sops_conv2 = n_spikes_conv2 * (5 * 32)
            
            layer_sops['conv2'].append(sops_conv2 / batch_size)
            layer_spike_rates['conv2'].append(spike_rate_conv2)
            
            # Conv3
            conv3_spikes = spikes_dict['conv3']
            n_spikes_conv3 = conv3_spikes.sum().item()
            spike_rate_conv3 = conv3_spikes.float().mean().item()
            # Conv3: in=64, kernel=3
            sops_conv3 = n_spikes_conv3 * (3 * 64)
            
            layer_sops['conv3'].append(sops_conv3 / batch_size)
            layer_spike_rates['conv3'].append(spike_rate_conv3)
            
            # FC1
            fc1_spikes = spikes_dict['fc1']
            n_spikes_fc1 = fc1_spikes.sum().item()
            spike_rate_fc1 = fc1_spikes.float().mean().item()
            # FC1: in=128*45=5760
            sops_fc1 = n_spikes_fc1 * 5760
            
            layer_sops['fc1'].append(sops_fc1 / batch_size)
            layer_spike_rates['fc1'].append(spike_rate_fc1)
            
            # FC2
            fc2_spikes = spikes_dict['fc2']
            n_spikes_fc2 = fc2_spikes.sum().item()
            spike_rate_fc2 = fc2_spikes.float().mean().item()
            # FC2: in=128
            sops_fc2 = n_spikes_fc2 * 128
            
            layer_sops['fc2'].append(sops_fc2 / batch_size)
            layer_spike_rates['fc2'].append(spike_rate_fc2)
    
    # Average across batches
    avg_sops = {k: np.mean(v) for k, v in layer_sops.items()}
    avg_spike_rates = {k: np.mean(v) for k, v in layer_spike_rates.items()}
    total_sops = sum(avg_sops.values())
    
    return avg_sops, avg_spike_rates, total_sops


def count_cnn_flops(model, input_size=360):
    """
    Count FLOPs for CNN.
    
    FLOPs definition:
    - Conv: 2 × (kernel_size × in_channels × out_channels) × output_size
    - FC: 2 × in_features × out_features
    """
    
    flops = {}
    
    # Conv1: in=1, out=32, kernel=5, input=360
    output_size_conv1 = input_size  # same padding
    flops['conv1'] = 2 * (5 * 1 * 32) * output_size_conv1
    
    # Pool1: output = 180
    output_size_pool1 = output_size_conv1 // 2
    
    # Conv2: in=32, out=64, kernel=5, input=180
    flops['conv2'] = 2 * (5 * 32 * 64) * output_size_pool1
    
    # Pool2: output = 90
    output_size_pool2 = output_size_pool1 // 2
    
    # Conv3: in=64, out=128, kernel=3, input=90
    flops['conv3'] = 2 * (3 * 64 * 128) * output_size_pool2
    
    # Pool3: output = 45
    output_size_pool3 = output_size_pool2 // 2
    
    # FC1: in=128*45=5760, out=256
    flops['fc1'] = 2 * 5760 * 256
    
    # FC2: in=256, out=2
    flops['fc2'] = 2 * 256 * 2
    
    total_flops = sum(flops.values())
    
    return flops, total_flops


def plot_sops_analysis(snn_sops, snn_spike_rates, cnn_flops, output_path):
    """Plot detailed SOPs vs FLOPs analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SOPs vs FLOPs Analysis: Layer-by-Layer Breakdown', fontsize=16, fontweight='bold')
    
    layers = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
    
    # Plot 1: SOPs per layer
    ax1 = axes[0, 0]
    snn_values = [snn_sops[l] for l in layers]
    ax1.bar(layers, snn_values, color='#2E86AB', alpha=0.7)
    ax1.set_ylabel('SOPs per Sample', fontsize=12)
    ax1.set_title('SNN: SOPs per Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(snn_values):
        ax1.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: FLOPs per layer
    ax2 = axes[0, 1]
    cnn_values = [cnn_flops[l] for l in layers]
    ax2.bar(layers, cnn_values, color='#A23B72', alpha=0.7)
    ax2.set_ylabel('FLOPs per Sample', fontsize=12)
    ax2.set_title('CNN: FLOPs per Layer', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(cnn_values):
        ax2.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Spike rates
    ax3 = axes[1, 0]
    spike_rate_values = [snn_spike_rates[l] for l in layers]
    ax3.bar(layers, spike_rate_values, color='#F18F01', alpha=0.7)
    ax3.set_ylabel('Spike Rate', fontsize=12)
    ax3.set_title('SNN: Spike Rate per Layer', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, max(spike_rate_values) * 1.2])
    for i, v in enumerate(spike_rate_values):
        ax3.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: SOPs/FLOPs ratio
    ax4 = axes[1, 1]
    ratios = [snn_sops[l] / cnn_flops[l] for l in layers]
    colors = ['green' if r < 1 else 'red' for r in ratios]
    ax4.bar(layers, ratios, color=colors, alpha=0.7)
    ax4.axhline(1.0, color='red', linestyle='--', linewidth=2, label='SOPs = FLOPs')
    ax4.set_ylabel('SOPs / FLOPs', fontsize=12)
    ax4.set_title('Efficiency Ratio (SOPs/FLOPs)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(ratios):
        ax4.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    config = load_config('/home/ubuntu/ecg_snn_project/config/config.yaml')
    set_seed(42)
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = '/home/ubuntu/ecg_snn_project/data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    import os
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(data_dir, existing_records, window_size=360, encoder=encoder)
    
    kfold = StratifiedPatientKFold(n_splits=5, random_state=42)
    splits = list(kfold.split(dataset))
    _, val_idx = splits[0]
    
    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=64, num_workers=0)
    
    # Load SNN
    snn = TemporalCSNN(input_size=360).to(device)
    snn.load_state_dict(torch.load(
        '/home/ubuntu/ecg_snn_project/models/snn_stable_fold0.pth',
        weights_only=True
    ))
    
    print("="*70)
    print("SOPs Detailed Analysis")
    print("="*70)
    
    # Count SOPs
    print("\nCounting SOPs for SNN...")
    snn_sops, snn_spike_rates, total_sops = count_snn_sops_detailed(snn, val_loader, device)
    
    print("\nSNN SOPs per Layer:")
    for layer, sops in snn_sops.items():
        print(f"  {layer:10s}: {sops:12.0f} SOPs (spike rate: {snn_spike_rates[layer]:.4f})")
    print(f"  {'TOTAL':10s}: {total_sops:12.0f} SOPs")
    
    # Count FLOPs
    print("\nCounting FLOPs for CNN...")
    cnn_flops, total_flops = count_cnn_flops(input_size=360)
    
    print("\nCNN FLOPs per Layer:")
    for layer, flops in cnn_flops.items():
        print(f"  {layer:10s}: {flops:12.0f} FLOPs")
    print(f"  {'TOTAL':10s}: {total_flops:12.0f} FLOPs")
    
    # Comparison
    print("\n" + "="*70)
    print("Comparison")
    print("="*70)
    
    ratio = total_sops / total_flops
    print(f"\nTotal SOPs: {total_sops:.0f}")
    print(f"Total FLOPs: {total_flops:.0f}")
    print(f"SOPs / FLOPs: {ratio:.4f}")
    
    if ratio <= 1.0:
        print(f"\n✓ SUCCESS: SOPs/FLOPs = {ratio:.4f} <= 1.0")
        print("  → SNN is more power-efficient than CNN")
    else:
        print(f"\n✗ FAIL: SOPs/FLOPs = {ratio:.4f} > 1.0")
        print("  → SNN is less efficient than CNN")
        print("\nPossible reasons:")
        print("  - High spike rates in early layers")
        print("  - Insufficient sparsity")
        print("\nSuggested improvements:")
        print("  - Increase LIF threshold to reduce spike rate")
        print("  - Add lateral inhibition")
        print("  - Use sparser encoding")
    
    # Plot
    plot_sops_analysis(snn_sops, snn_spike_rates, cnn_flops,
                      '/home/ubuntu/ecg_snn_project/results/sops_detailed_analysis.png')
    
    print("\nResults saved to: /home/ubuntu/ecg_snn_project/results/sops_detailed_analysis.png")
