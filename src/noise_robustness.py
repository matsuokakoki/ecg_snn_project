"""
Noise robustness evaluation for wearable ECG applications.
Tests: Gaussian noise, baseline wander, amplitude variation.
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

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN, BaselineCNN


class NoiseAugmentation:
    """Noise augmentation for ECG signals."""
    
    @staticmethod
    def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Gaussian noise with specified SNR."""
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    @staticmethod
    def add_baseline_wander(signal: np.ndarray, amplitude: float, frequency: float = 0.5, fs: float = 360) -> np.ndarray:
        """Add sinusoidal baseline wander."""
        t = np.arange(len(signal)) / fs
        wander = amplitude * np.sin(2 * np.pi * frequency * t)
        return signal + wander
    
    @staticmethod
    def add_amplitude_variation(signal: np.ndarray, variation_factor: float) -> np.ndarray:
        """Add random amplitude scaling."""
        scale = 1.0 + np.random.uniform(-variation_factor, variation_factor)
        return signal * scale
    
    @staticmethod
    def add_motion_artifact(signal: np.ndarray, artifact_prob: float = 0.1, artifact_amplitude: float = 0.5) -> np.ndarray:
        """Add random motion artifacts (sudden spikes)."""
        noisy = signal.copy()
        artifact_mask = np.random.random(len(signal)) < artifact_prob
        artifacts = np.random.uniform(-artifact_amplitude, artifact_amplitude, np.sum(artifact_mask))
        noisy[artifact_mask] += artifacts * np.std(signal)
        return noisy


def evaluate_with_noise(model, data_loader, device, noise_fn=None, is_snn=False):
    """Evaluate model with optional noise augmentation."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in data_loader:
            # Apply noise if specified
            if noise_fn is not None:
                data_np = data.numpy()
                noisy_data = np.array([noise_fn(d) for d in data_np])
                data = torch.tensor(noisy_data, dtype=torch.float32)
            
            data = data.to(device)
            
            if is_snn:
                _, output = model(data)
            else:
                output = model(data)
            
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return MetricsCalculator.calculate_all(np.array(all_targets), np.array(all_preds), np.array(all_probs))


def main():
    config = load_config('./config/config.yaml')
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=config['encoding']['num_levels'], threshold_factor=config['encoding']['threshold_factor'])
    dataset = MITBIHDataset(data_dir, existing_records, window_size=config['data']['window_size'], encoder=encoder)
    
    # Use test set
    kfold = StratifiedPatientKFold(n_splits=5, random_state=config['experiment']['seed'])
    train_idx, test_idx = next(kfold.split(dataset))
    
    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=64)
    
    # Load models
    snn_model = TemporalCSNN(input_size=config['data']['window_size']).to(device)
    cnn_model = BaselineCNN(input_size=config['data']['window_size']).to(device)
    
    snn_path = './models/temporal_csnn_v2_fold0_best.pth'
    cnn_path = './models/baseline_cnn_v2_fold0_best.pth'
    
    if os.path.exists(snn_path):
        snn_model.load_state_dict(torch.load(snn_path, weights_only=True))
    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, weights_only=True))
    
    print("="*70)
    print("Noise Robustness Evaluation")
    print("="*70)
    
    results = {
        'snn': {'clean': None, 'gaussian': {}, 'baseline': {}, 'amplitude': {}},
        'cnn': {'clean': None, 'gaussian': {}, 'baseline': {}, 'amplitude': {}}
    }
    
    # Clean evaluation
    print("\n--- Clean Data ---")
    results['snn']['clean'] = evaluate_with_noise(snn_model, test_loader, device, is_snn=True)
    results['cnn']['clean'] = evaluate_with_noise(cnn_model, test_loader, device, is_snn=False)
    print(f"SNN: Acc={results['snn']['clean']['accuracy']:.4f}, F1={results['snn']['clean']['macro_f1']:.4f}")
    print(f"CNN: Acc={results['cnn']['clean']['accuracy']:.4f}, F1={results['cnn']['clean']['macro_f1']:.4f}")
    
    # Gaussian noise test
    print("\n--- Gaussian Noise Test ---")
    snr_levels = [30, 20, 10, 5]  # dB
    
    for snr in snr_levels:
        noise_fn = lambda x, s=snr: NoiseAugmentation.add_gaussian_noise(x, s)
        results['snn']['gaussian'][snr] = evaluate_with_noise(snn_model, test_loader, device, noise_fn, is_snn=True)
        results['cnn']['gaussian'][snr] = evaluate_with_noise(cnn_model, test_loader, device, noise_fn, is_snn=False)
        print(f"SNR={snr}dB: SNN Acc={results['snn']['gaussian'][snr]['accuracy']:.4f}, "
              f"CNN Acc={results['cnn']['gaussian'][snr]['accuracy']:.4f}")
    
    # Baseline wander test
    print("\n--- Baseline Wander Test ---")
    wander_amplitudes = [0.1, 0.2, 0.3, 0.5]
    
    for amp in wander_amplitudes:
        noise_fn = lambda x, a=amp: NoiseAugmentation.add_baseline_wander(x, a)
        results['snn']['baseline'][amp] = evaluate_with_noise(snn_model, test_loader, device, noise_fn, is_snn=True)
        results['cnn']['baseline'][amp] = evaluate_with_noise(cnn_model, test_loader, device, noise_fn, is_snn=False)
        print(f"Amp={amp}: SNN Acc={results['snn']['baseline'][amp]['accuracy']:.4f}, "
              f"CNN Acc={results['cnn']['baseline'][amp]['accuracy']:.4f}")
    
    # Amplitude variation test
    print("\n--- Amplitude Variation Test ---")
    variation_factors = [0.1, 0.2, 0.3, 0.5]
    
    for var in variation_factors:
        noise_fn = lambda x, v=var: NoiseAugmentation.add_amplitude_variation(x, v)
        results['snn']['amplitude'][var] = evaluate_with_noise(snn_model, test_loader, device, noise_fn, is_snn=True)
        results['cnn']['amplitude'][var] = evaluate_with_noise(cnn_model, test_loader, device, noise_fn, is_snn=False)
        print(f"Var={var}: SNN Acc={results['snn']['amplitude'][var]['accuracy']:.4f}, "
              f"CNN Acc={results['cnn']['amplitude'][var]['accuracy']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gaussian noise
    snr_labels = ['Clean'] + [f'{s}dB' for s in snr_levels]
    snn_acc_gauss = [results['snn']['clean']['accuracy']] + [results['snn']['gaussian'][s]['accuracy'] for s in snr_levels]
    cnn_acc_gauss = [results['cnn']['clean']['accuracy']] + [results['cnn']['gaussian'][s]['accuracy'] for s in snr_levels]
    
    axes[0, 0].plot(snr_labels, snn_acc_gauss, 'b-o', label='SNN', linewidth=2, markersize=8)
    axes[0, 0].plot(snr_labels, cnn_acc_gauss, 'r-s', label='CNN', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('SNR Level')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Gaussian Noise Robustness')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Baseline wander
    wander_labels = ['Clean'] + [f'{a}' for a in wander_amplitudes]
    snn_acc_wander = [results['snn']['clean']['accuracy']] + [results['snn']['baseline'][a]['accuracy'] for a in wander_amplitudes]
    cnn_acc_wander = [results['cnn']['clean']['accuracy']] + [results['cnn']['baseline'][a]['accuracy'] for a in wander_amplitudes]
    
    axes[0, 1].plot(wander_labels, snn_acc_wander, 'b-o', label='SNN', linewidth=2, markersize=8)
    axes[0, 1].plot(wander_labels, cnn_acc_wander, 'r-s', label='CNN', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Wander Amplitude')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Baseline Wander Robustness')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Amplitude variation
    var_labels = ['Clean'] + [f'{v}' for v in variation_factors]
    snn_acc_var = [results['snn']['clean']['accuracy']] + [results['snn']['amplitude'][v]['accuracy'] for v in variation_factors]
    cnn_acc_var = [results['cnn']['clean']['accuracy']] + [results['cnn']['amplitude'][v]['accuracy'] for v in variation_factors]
    
    axes[1, 0].plot(var_labels, snn_acc_var, 'b-o', label='SNN', linewidth=2, markersize=8)
    axes[1, 0].plot(var_labels, cnn_acc_var, 'r-s', label='CNN', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Variation Factor')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Amplitude Variation Robustness')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary bar chart
    noise_types = ['Gaussian\n(10dB)', 'Baseline\n(0.3)', 'Amplitude\n(0.3)']
    snn_degradation = [
        results['snn']['clean']['accuracy'] - results['snn']['gaussian'][10]['accuracy'],
        results['snn']['clean']['accuracy'] - results['snn']['baseline'][0.3]['accuracy'],
        results['snn']['clean']['accuracy'] - results['snn']['amplitude'][0.3]['accuracy']
    ]
    cnn_degradation = [
        results['cnn']['clean']['accuracy'] - results['cnn']['gaussian'][10]['accuracy'],
        results['cnn']['clean']['accuracy'] - results['cnn']['baseline'][0.3]['accuracy'],
        results['cnn']['clean']['accuracy'] - results['cnn']['amplitude'][0.3]['accuracy']
    ]
    
    x = np.arange(len(noise_types))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, snn_degradation, width, label='SNN', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, cnn_degradation, width, label='CNN', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Noise Type')
    axes[1, 1].set_ylabel('Accuracy Degradation')
    axes[1, 1].set_title('Performance Degradation (Lower is Better)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(noise_types)
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('./results/noise_robustness.png', dpi=150)
    print("\nSaved to results/noise_robustness.png")
    
    # Summary
    print("\n" + "="*70)
    print("Robustness Summary")
    print("="*70)
    print(f"\nAccuracy Degradation at Moderate Noise:")
    print(f"  Gaussian (10dB): SNN={snn_degradation[0]:.4f}, CNN={cnn_degradation[0]:.4f}")
    print(f"  Baseline (0.3):  SNN={snn_degradation[1]:.4f}, CNN={cnn_degradation[1]:.4f}")
    print(f"  Amplitude (0.3): SNN={snn_degradation[2]:.4f}, CNN={cnn_degradation[2]:.4f}")
    
    return results


if __name__ == "__main__":
    main()
