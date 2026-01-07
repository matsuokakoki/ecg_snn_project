"""
Visualization of LIF neuron behavior and spike encoding.
Includes ISI analysis and temporal coding visualization.
"""
import sys
sys.path.insert(0, '/home/ubuntu/ecg_snn_project/src')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# Set font for Japanese support
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

import torch
import snntorch as snn
from snntorch import surrogate
import wfdb
from data_loader import GradedDeltaEncoder

def visualize_lif_dynamics():
    """Visualize LIF neuron membrane potential and spike dynamics."""
    print("Visualizing LIF neuron dynamics...")
    
    # Parameters
    beta = 0.9  # Membrane decay
    num_steps = 100
    spike_grad = surrogate.fast_sigmoid(slope=25)
    
    # Create LIF neuron
    lif = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    # Different input currents to show behavior
    input_patterns = {
        'Constant': torch.cat([torch.zeros(10), torch.ones(60)*0.5, torch.zeros(30)]),
        'Ramp': torch.cat([torch.zeros(10), torch.linspace(0, 1, 60), torch.zeros(30)]),
        'Pulse': torch.cat([torch.zeros(20), torch.ones(10)*1.5, torch.zeros(70)])
    }
    
    fig, axes = plt.subplots(len(input_patterns), 2, figsize=(14, 10))
    
    for idx, (name, cur_in) in enumerate(input_patterns.items()):
        mem = torch.zeros(1)
        spk_out = []
        mem_rec = []
        
        for step in range(num_steps):
            spk, mem = lif(cur_in[step], mem)
            spk_out.append(spk.item())
            mem_rec.append(mem.item())
        
        # Plot membrane potential
        axes[idx, 0].plot(mem_rec, 'b-', linewidth=1.5, label='Membrane Potential')
        axes[idx, 0].axhline(y=1.0, color='r', linestyle='--', label='Threshold')
        axes[idx, 0].fill_between(range(num_steps), 0, cur_in.numpy()*0.5, alpha=0.3, label='Input Current')
        axes[idx, 0].set_ylabel('Voltage')
        axes[idx, 0].set_title(f'LIF Dynamics: {name} Input')
        axes[idx, 0].legend(loc='upper right')
        axes[idx, 0].set_ylim(-0.1, 1.5)
        
        # Plot spikes
        spike_times = np.where(np.array(spk_out) > 0)[0]
        axes[idx, 1].eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='blue')
        axes[idx, 1].set_ylabel('Spikes')
        axes[idx, 1].set_xlabel('Time Step')
        axes[idx, 1].set_title(f'Spike Train: {len(spike_times)} spikes')
        axes[idx, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/ecg_snn_project/results/lif_dynamics.png', dpi=150)
    print("Saved: results/lif_dynamics.png")


def visualize_ecg_encoding():
    """Visualize ECG signal encoding with graded delta modulation."""
    print("Visualizing ECG encoding...")
    
    # Load sample ECG
    record = wfdb.rdrecord('/home/ubuntu/ecg_snn_project/data/mitdb/100')
    signal = record.p_signal[:3600, 0]  # 10 seconds
    fs = record.fs
    
    # Encode with different level settings
    encoders = {
        'Binary (L=1)': GradedDeltaEncoder(num_levels=1, threshold_factor=0.5),
        'Graded (L=3)': GradedDeltaEncoder(num_levels=3, threshold_factor=0.5),
        'Graded (L=5)': GradedDeltaEncoder(num_levels=5, threshold_factor=0.5)
    }
    
    fig, axes = plt.subplots(len(encoders) + 1, 1, figsize=(14, 12), sharex=True)
    time = np.arange(len(signal)) / fs
    
    # Original signal
    axes[0].plot(time, signal, 'k-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original ECG Signal (MIT-BIH Record 100)')
    
    # Encoded signals
    for idx, (name, encoder) in enumerate(encoders.items()):
        spikes = encoder.encode(signal)
        
        # Plot spikes with color coding for magnitude
        pos_mask = spikes > 0
        neg_mask = spikes < 0
        
        axes[idx+1].stem(time[pos_mask], spikes[pos_mask], 'b', markerfmt=' ', basefmt=' ', label='Positive')
        axes[idx+1].stem(time[neg_mask], spikes[neg_mask], 'r', markerfmt=' ', basefmt=' ', label='Negative')
        axes[idx+1].set_ylabel('Spike Magnitude')
        axes[idx+1].set_title(f'{name}: {np.sum(spikes != 0)} spikes (Rate: {np.sum(spikes != 0)/len(signal)*100:.2f}%)')
        axes[idx+1].legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/ecg_snn_project/results/ecg_encoding_comparison.png', dpi=150)
    print("Saved: results/ecg_encoding_comparison.png")


def visualize_isi_distribution():
    """Visualize Inter-Spike Interval distribution."""
    print("Visualizing ISI distribution...")
    
    # Load ECG and encode
    record = wfdb.rdrecord('/home/ubuntu/ecg_snn_project/data/mitdb/100')
    signal = record.p_signal[:36000, 0]  # 100 seconds for better statistics
    fs = record.fs
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    spikes = encoder.encode(signal)
    
    # Compute ISI
    spike_times = np.where(spikes != 0)[0]
    isi = np.diff(spike_times)
    
    # Also compute ISI for positive and negative spikes separately
    pos_times = np.where(spikes > 0)[0]
    neg_times = np.where(spikes < 0)[0]
    pos_isi = np.diff(pos_times) if len(pos_times) > 1 else np.array([])
    neg_isi = np.diff(neg_times) if len(neg_times) > 1 else np.array([])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall ISI histogram
    axes[0, 0].hist(isi, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('ISI (samples)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Overall ISI Distribution (Mean: {np.mean(isi):.1f}, Std: {np.std(isi):.1f})')
    axes[0, 0].axvline(np.mean(isi), color='r', linestyle='--', label=f'Mean={np.mean(isi):.1f}')
    axes[0, 0].legend()
    
    # Positive spike ISI
    if len(pos_isi) > 0:
        axes[0, 1].hist(pos_isi, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('ISI (samples)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Positive Spike ISI (Mean: {np.mean(pos_isi):.1f})')
    
    # Negative spike ISI
    if len(neg_isi) > 0:
        axes[1, 0].hist(neg_isi, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('ISI (samples)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Negative Spike ISI (Mean: {np.mean(neg_isi):.1f})')
    
    # Spike timing around R-wave (zoom into one beat)
    # Find a segment with clear R-wave
    segment_start = 1000
    segment_end = 1360
    segment_signal = signal[segment_start:segment_end]
    segment_spikes = spikes[segment_start:segment_end]
    segment_time = np.arange(len(segment_signal)) / fs * 1000  # ms
    
    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(segment_time, segment_signal, 'k-', linewidth=1, label='ECG')
    
    spike_idx = np.where(segment_spikes != 0)[0]
    ax2.stem(segment_time[spike_idx], segment_spikes[spike_idx], 'b', markerfmt='o', basefmt=' ', label='Spikes')
    
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('ECG Amplitude')
    ax2.set_ylabel('Spike Magnitude')
    axes[1, 1].set_title('Spike Timing Around R-wave')
    axes[1, 1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/ecg_snn_project/results/isi_distribution.png', dpi=150)
    print("Saved: results/isi_distribution.png")


def visualize_firing_rate_per_beat():
    """Visualize firing rate statistics per beat."""
    print("Visualizing firing rate per beat...")
    
    from data_loader import MITBIHDataset
    
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(
        '/home/ubuntu/ecg_snn_project/data/mitdb',
        ['100', '101'],
        encoder=encoder
    )
    
    # Calculate firing rates
    normal_rates = []
    abnormal_rates = []
    
    for i in range(len(dataset)):
        spikes = dataset.spike_data[i]
        rate = np.sum(spikes != 0) / len(spikes)
        
        if dataset.labels[i] == 0:
            normal_rates.append(rate)
        else:
            abnormal_rates.append(rate)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram comparison
    axes[0].hist(normal_rates, bins=30, alpha=0.7, label=f'Normal (n={len(normal_rates)})', color='blue')
    axes[0].hist(abnormal_rates, bins=30, alpha=0.7, label=f'Abnormal (n={len(abnormal_rates)})', color='red')
    axes[0].set_xlabel('Firing Rate (spikes/sample)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Firing Rate Distribution by Class')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot([normal_rates, abnormal_rates], labels=['Normal', 'Abnormal'])
    axes[1].set_ylabel('Firing Rate')
    axes[1].set_title('Firing Rate Comparison')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/ecg_snn_project/results/firing_rate_per_beat.png', dpi=150)
    print("Saved: results/firing_rate_per_beat.png")
    
    print(f"\nFiring Rate Statistics:")
    print(f"  Normal: Mean={np.mean(normal_rates):.4f}, Std={np.std(normal_rates):.4f}")
    print(f"  Abnormal: Mean={np.mean(abnormal_rates):.4f}, Std={np.std(abnormal_rates):.4f}")


if __name__ == "__main__":
    visualize_lif_dynamics()
    visualize_ecg_encoding()
    visualize_isi_distribution()
    visualize_firing_rate_per_beat()
    print("\nAll visualizations complete!")
