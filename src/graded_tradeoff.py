"""
Graded spike tradeoff analysis and implementation outlook.
Analyzes the tradeoff between spike resolution (L levels) and:
- Accuracy
- Sparsity
- Hardware complexity
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

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN


def analyze_encoding_levels():
    """Analyze the effect of different graded spike levels."""
    config = load_config('/home/ubuntu/ecg_snn_project/config/config.yaml')
    set_seed(config['experiment']['seed'])
    
    data_dir = '/home/ubuntu/ecg_snn_project/data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    levels_to_test = [1, 2, 3, 4, 5]
    results = {}
    
    print("="*60)
    print("Graded Spike Level Analysis")
    print("="*60)
    
    for num_levels in levels_to_test:
        print(f"\n--- Testing L={num_levels} ---")
        
        encoder = GradedDeltaEncoder(num_levels=num_levels, threshold_factor=0.5)
        dataset = MITBIHDataset(data_dir, existing_records, window_size=config['data']['window_size'], encoder=encoder)
        
        # Calculate sparsity and spike statistics
        total_spikes = 0
        total_samples = 0
        spike_magnitudes = []
        
        for i in range(min(1000, len(dataset))):
            spikes = dataset.spike_data[i]
            total_spikes += np.sum(np.abs(spikes) > 0)
            total_samples += len(spikes)
            spike_magnitudes.extend(np.abs(spikes[spikes != 0]).tolist())
        
        sparsity = 1 - (total_spikes / total_samples)
        avg_magnitude = np.mean(spike_magnitudes) if spike_magnitudes else 0
        
        results[num_levels] = {
            'sparsity': sparsity,
            'avg_magnitude': avg_magnitude,
            'spike_rate': total_spikes / total_samples
        }
        
        print(f"  Sparsity: {sparsity:.4f}")
        print(f"  Spike Rate: {results[num_levels]['spike_rate']:.4f}")
        print(f"  Avg Magnitude: {avg_magnitude:.2f}")
    
    return results


def hardware_complexity_analysis():
    """Analyze hardware implementation complexity for different configurations."""
    
    print("\n" + "="*60)
    print("Hardware Implementation Analysis")
    print("="*60)
    
    # Model parameters
    model = TemporalCSNN(input_size=360)
    params = model.count_parameters()
    
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {params:,}")
    print(f"  Memory (32-bit): {params * 4 / 1024:.2f} KB")
    print(f"  Memory (8-bit quantized): {params / 1024:.2f} KB")
    
    # Estimate for different hardware targets
    hardware_targets = {
        'ESP32': {'ram_kb': 520, 'flash_mb': 4, 'clock_mhz': 240},
        'STM32F4': {'ram_kb': 192, 'flash_mb': 1, 'clock_mhz': 168},
        'nRF52840': {'ram_kb': 256, 'flash_mb': 1, 'clock_mhz': 64},
        'Loihi 2': {'neurons': 1000000, 'synapses': 120000000, 'power_mw': 1000},
        'Akida': {'neurons': 1200000, 'synapses': 10000000000, 'power_mw': 300}
    }
    
    print("\n" + "-"*60)
    print("Target Hardware Compatibility")
    print("-"*60)
    
    model_ram_kb = params / 1024  # 8-bit quantized
    
    for hw, specs in hardware_targets.items():
        if 'ram_kb' in specs:
            fits = "✓" if model_ram_kb < specs['ram_kb'] * 0.5 else "△" if model_ram_kb < specs['ram_kb'] else "✗"
            print(f"\n{hw}:")
            print(f"  RAM: {specs['ram_kb']} KB (Model: {model_ram_kb:.1f} KB) {fits}")
            print(f"  Flash: {specs['flash_mb']} MB")
            print(f"  Clock: {specs['clock_mhz']} MHz")
        else:
            print(f"\n{hw} (Neuromorphic):")
            print(f"  Neurons: {specs['neurons']:,}")
            print(f"  Synapses: {specs['synapses']:,}")
            print(f"  Power: {specs['power_mw']} mW")
    
    # Energy estimation
    print("\n" + "-"*60)
    print("Energy Estimation")
    print("-"*60)
    
    # Assumptions
    sops_per_inference = 23e6  # From our measurements
    energy_per_sop_pj = 1  # Typical for neuromorphic
    energy_per_flop_pj = 10  # Typical for MCU
    
    cnn_flops = 1054336
    
    snn_energy_uj = sops_per_inference * energy_per_sop_pj / 1e6
    cnn_energy_uj = cnn_flops * energy_per_flop_pj / 1e6
    
    print(f"\nPer-Inference Energy (estimated):")
    print(f"  SNN on Neuromorphic: {snn_energy_uj:.2f} µJ")
    print(f"  CNN on MCU: {cnn_energy_uj:.2f} µJ")
    print(f"  Ratio: {cnn_energy_uj/snn_energy_uj:.1f}x")
    
    # Battery life estimation
    battery_mah = 100  # Small wearable battery
    battery_wh = battery_mah * 3.7 / 1000
    battery_j = battery_wh * 3600
    
    inferences_per_second = 1  # 1 beat per second
    
    snn_runtime_hours = battery_j / (snn_energy_uj * 1e-6 * inferences_per_second * 3600)
    cnn_runtime_hours = battery_j / (cnn_energy_uj * 1e-6 * inferences_per_second * 3600)
    
    print(f"\nBattery Life (100mAh, 1 inference/sec):")
    print(f"  SNN: {snn_runtime_hours:.0f} hours ({snn_runtime_hours/24:.1f} days)")
    print(f"  CNN: {cnn_runtime_hours:.0f} hours ({cnn_runtime_hours/24:.1f} days)")
    
    return {
        'params': params,
        'snn_energy_uj': snn_energy_uj,
        'cnn_energy_uj': cnn_energy_uj,
        'snn_runtime_hours': snn_runtime_hours,
        'cnn_runtime_hours': cnn_runtime_hours
    }


def create_summary_visualization(encoding_results, hw_results):
    """Create summary visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sparsity vs Levels
    levels = list(encoding_results.keys())
    sparsities = [encoding_results[l]['sparsity'] for l in levels]
    spike_rates = [encoding_results[l]['spike_rate'] for l in levels]
    
    axes[0, 0].bar(levels, sparsities, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Graded Levels (L)')
    axes[0, 0].set_ylabel('Sparsity')
    axes[0, 0].set_title('Sparsity vs Graded Spike Levels')
    axes[0, 0].set_ylim(0.9, 1.0)
    
    # Spike rate vs Levels
    axes[0, 1].bar(levels, spike_rates, color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Graded Levels (L)')
    axes[0, 1].set_ylabel('Spike Rate')
    axes[0, 1].set_title('Spike Rate vs Graded Spike Levels')
    
    # Energy comparison
    models = ['SNN\n(Neuromorphic)', 'CNN\n(MCU)']
    energies = [hw_results['snn_energy_uj'], hw_results['cnn_energy_uj']]
    colors = ['blue', 'red']
    
    axes[1, 0].bar(models, energies, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Energy per Inference (µJ)')
    axes[1, 0].set_title('Energy Consumption Comparison')
    axes[1, 0].set_yscale('log')
    
    # Battery life comparison
    runtimes = [hw_results['snn_runtime_hours']/24, hw_results['cnn_runtime_hours']/24]
    
    axes[1, 1].bar(models, runtimes, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Battery Life (days)')
    axes[1, 1].set_title('Estimated Battery Life (100mAh)')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/ecg_snn_project/results/graded_tradeoff.png', dpi=150)
    print("\nSaved to results/graded_tradeoff.png")


def main():
    # Encoding level analysis
    encoding_results = analyze_encoding_levels()
    
    # Hardware analysis
    hw_results = hardware_complexity_analysis()
    
    # Create visualization
    create_summary_visualization(encoding_results, hw_results)
    
    # Implementation roadmap
    print("\n" + "="*60)
    print("Implementation Roadmap")
    print("="*60)
    
    print("""
Phase 1: PC Simulation (Current)
  ✓ snntorch-based SNN implementation
  ✓ Graded delta modulation encoding
  ✓ STDP personalization simulation
  ✓ SOPs measurement framework

Phase 2: MCU Deployment (Near-term)
  - Quantize model to 8-bit integers
  - Port to ESP32/STM32 using TensorFlow Lite Micro
  - Implement delta encoding in C
  - Target: <100ms inference, <1mW average power

Phase 3: Neuromorphic Chip (Future)
  - Convert to Lava framework (Intel Loihi 2)
  - Or MetaTF (BrainChip Akida)
  - Enable on-chip STDP learning
  - Target: <10µW continuous operation

Key Advantages of SNN Approach:
  1. Event-driven computation (no activity = no power)
  2. Temporal coding preserves timing information
  3. On-chip learning enables personalization
  4. Compatible with emerging neuromorphic hardware
""")


if __name__ == "__main__":
    main()
