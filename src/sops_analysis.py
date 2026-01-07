"""
Strict SOPs (Synaptic Operations) measurement and analysis.
SOPs = spike_count × fan_out (number of outgoing synapses)
"""
import sys
sys.path.insert(0, '/home/ubuntu/ecg_snn_project/src')

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import set_seed, load_config
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN


class SOPsAnalyzer:
    """Analyze Synaptic Operations in SNN."""
    
    def __init__(self, model):
        self.model = model
        self.layer_stats = {}
    
    def analyze_batch(self, data):
        """Analyze SOPs for a batch of data."""
        self.model.eval()
        self.model.enable_spike_tracking()
        
        with torch.no_grad():
            _, _ = self.model(data)
        
        spike_counts = self.model.get_spike_counts()
        batch_size = data.size(0)
        
        # Calculate SOPs per layer
        # SOPs = spike_count × fan_out
        sops = {}
        
        # Layer 1: spikes go to layer 2 (fc_input_size connections per spike)
        sops['layer1'] = {
            'spike_count': spike_counts['layer1'],
            'fan_out': self.model.fc_input_size,
            'sops': spike_counts['layer1'] * self.model.fc_input_size
        }
        
        # Layer 2: spikes go to FC layer (64 connections per spike)
        sops['layer2'] = {
            'spike_count': spike_counts['layer2'],
            'fan_out': 64,  # FC output size
            'sops': spike_counts['layer2'] * 64
        }
        
        # FC layer: spikes go to output (2 connections per spike)
        sops['fc'] = {
            'spike_count': spike_counts['fc'],
            'fan_out': 2,
            'sops': spike_counts['fc'] * 2
        }
        
        # Total
        sops['total'] = {
            'spike_count': sum(s['spike_count'] for s in sops.values() if isinstance(s, dict)),
            'sops': sum(s['sops'] for s in sops.values() if isinstance(s, dict))
        }
        
        # Per sample
        for layer in sops:
            if isinstance(sops[layer], dict):
                sops[layer]['sops_per_sample'] = sops[layer]['sops'] / batch_size
                sops[layer]['spikes_per_sample'] = sops[layer]['spike_count'] / batch_size
        
        return sops
    
    def analyze_dataset(self, data_loader):
        """Analyze SOPs across entire dataset."""
        all_sops = []
        all_spikes = []
        
        for data, _ in data_loader:
            sops = self.analyze_batch(data)
            all_sops.append(sops['total']['sops_per_sample'])
            all_spikes.append(sops['total']['spikes_per_sample'])
        
        return {
            'mean_sops': np.mean(all_sops),
            'std_sops': np.std(all_sops),
            'mean_spikes': np.mean(all_spikes),
            'std_spikes': np.std(all_spikes),
            'min_sops': np.min(all_sops),
            'max_sops': np.max(all_sops)
        }


def calculate_firing_rate(model, data_loader, device):
    """Calculate firing rate per layer."""
    model.eval()
    model.enable_spike_tracking()
    
    total_spikes = {'layer1': 0, 'layer2': 0, 'fc': 0}
    total_neurons_time = {'layer1': 0, 'layer2': 0, 'fc': 0}
    
    num_steps = model.num_steps
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            batch_size = data.size(0)
            
            _, _ = model(data)
            spike_counts = model.get_spike_counts()
            
            # Layer 1: fc_input_size neurons × num_steps × batch_size
            total_spikes['layer1'] += spike_counts['layer1']
            total_neurons_time['layer1'] += model.fc_input_size * num_steps * batch_size
            
            # Layer 2: fc_input_size neurons
            total_spikes['layer2'] += spike_counts['layer2']
            total_neurons_time['layer2'] += model.fc_input_size * num_steps * batch_size
            
            # FC: 64 neurons
            total_spikes['fc'] += spike_counts['fc']
            total_neurons_time['fc'] += 64 * num_steps * batch_size
    
    firing_rates = {}
    for layer in total_spikes:
        if total_neurons_time[layer] > 0:
            firing_rates[layer] = total_spikes[layer] / total_neurons_time[layer]
        else:
            firing_rates[layer] = 0
    
    return firing_rates


def main():
    config = load_config('/home/ubuntu/ecg_snn_project/config/config.yaml')
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = '/home/ubuntu/ecg_snn_project/data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=config['encoding']['num_levels'], threshold_factor=config['encoding']['threshold_factor'])
    dataset = MITBIHDataset(data_dir, existing_records, window_size=config['data']['window_size'], encoder=encoder)
    
    # Use test set
    kfold = StratifiedPatientKFold(n_splits=5, random_state=config['experiment']['seed'])
    train_idx, test_idx = next(kfold.split(dataset))
    
    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=64)
    
    # Load model
    model = TemporalCSNN(input_size=config['data']['window_size']).to(device)
    model_path = '/home/ubuntu/ecg_snn_project/models/temporal_csnn_v2_fold0_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded trained model")
    
    # Analyze SOPs
    print("\n" + "="*60)
    print("SOPs Analysis")
    print("="*60)
    
    analyzer = SOPsAnalyzer(model)
    
    # Single batch analysis
    data, _ = next(iter(test_loader))
    data = data.to(device)
    batch_sops = analyzer.analyze_batch(data)
    
    print("\nPer-Layer SOPs Breakdown (single batch):")
    print("-"*60)
    print(f"{'Layer':<15} {'Spikes':<15} {'Fan-out':<15} {'SOPs':<15}")
    print("-"*60)
    for layer in ['layer1', 'layer2', 'fc']:
        info = batch_sops[layer]
        print(f"{layer:<15} {info['spikes_per_sample']:.2f}{'':>7} {info['fan_out']:<15} {info['sops_per_sample']:.2f}")
    print("-"*60)
    print(f"{'Total':<15} {batch_sops['total']['spikes_per_sample']:.2f}{'':>7} {'-':<15} {batch_sops['total']['sops_per_sample']:.2f}")
    
    # Dataset-wide analysis
    dataset_stats = analyzer.analyze_dataset(test_loader)
    
    print(f"\nDataset-wide Statistics:")
    print(f"  Mean SOPs/sample: {dataset_stats['mean_sops']:.2f} ± {dataset_stats['std_sops']:.2f}")
    print(f"  Mean Spikes/sample: {dataset_stats['mean_spikes']:.2f} ± {dataset_stats['std_spikes']:.2f}")
    print(f"  Range: [{dataset_stats['min_sops']:.2f}, {dataset_stats['max_sops']:.2f}]")
    
    # Firing rates
    print("\n" + "="*60)
    print("Firing Rate Analysis")
    print("="*60)
    
    firing_rates = calculate_firing_rate(model, test_loader, device)
    
    print("\nFiring Rate per Layer:")
    for layer, rate in firing_rates.items():
        print(f"  {layer}: {rate:.4f} ({rate*100:.2f}%)")
    
    avg_firing_rate = np.mean(list(firing_rates.values()))
    print(f"\nAverage Firing Rate: {avg_firing_rate:.4f} ({avg_firing_rate*100:.2f}%)")
    
    # Energy efficiency discussion
    print("\n" + "="*60)
    print("Energy Efficiency Analysis")
    print("="*60)
    
    # Baseline CNN FLOPs
    from snn_model_v2 import BaselineCNN
    cnn = BaselineCNN(input_size=config['data']['window_size'])
    cnn_flops = cnn.count_flops()
    
    print(f"\nComparison:")
    print(f"  SNN SOPs/sample: {dataset_stats['mean_sops']:.0f}")
    print(f"  CNN FLOPs/sample: {cnn_flops}")
    
    # Note: Direct comparison is not straightforward
    # SOPs on neuromorphic hardware are much more energy-efficient than FLOPs on conventional hardware
    # Typical estimates: 1 SOP ≈ 0.1-10 pJ on neuromorphic chips, 1 FLOP ≈ 1-10 pJ on GPUs
    
    print(f"\nEnergy Efficiency Notes:")
    print(f"  - SOPs on neuromorphic hardware: ~0.1-10 pJ per operation")
    print(f"  - FLOPs on conventional hardware: ~1-10 pJ per operation")
    print(f"  - Low firing rate ({avg_firing_rate*100:.2f}%) indicates sparse computation")
    print(f"  - Sparse computation enables event-driven processing")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # SOPs by layer
    layers = ['Layer 1', 'Layer 2', 'FC']
    sops_values = [batch_sops['layer1']['sops_per_sample'], 
                   batch_sops['layer2']['sops_per_sample'],
                   batch_sops['fc']['sops_per_sample']]
    
    axes[0].bar(layers, sops_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('SOPs per sample')
    axes[0].set_title('SOPs Distribution by Layer')
    axes[0].set_yscale('log')
    
    # Firing rates
    axes[1].bar(layers, [firing_rates['layer1'], firing_rates['layer2'], firing_rates['fc']], 
                color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Firing Rate')
    axes[1].set_title('Firing Rate by Layer')
    axes[1].set_ylim(0, 1)
    
    # Sparsity visualization
    sparsity = [1 - r for r in [firing_rates['layer1'], firing_rates['layer2'], firing_rates['fc']]]
    axes[2].bar(layers, sparsity, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('Sparsity (1 - Firing Rate)')
    axes[2].set_title('Sparsity by Layer')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/ecg_snn_project/results/sops_analysis.png', dpi=150)
    print("\nSaved analysis plot to results/sops_analysis.png")


if __name__ == "__main__":
    main()
