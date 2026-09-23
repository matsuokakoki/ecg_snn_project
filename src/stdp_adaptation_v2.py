"""
Improved STDP adaptation with stability constraints.
"""
import sys
sys.path.insert(0, './src')

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import set_seed, load_config, MetricsCalculator
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN


class StableSTDPLearner:
    """
    Stable STDP with weight normalization and bounded updates.
    """
    
    def __init__(self, lr: float = 0.001, weight_decay: float = 0.01):
        self.lr = lr
        self.weight_decay = weight_decay
        self.trace = None
    
    def compute_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian-like weight update."""
        # Simple Hebbian: Δw ∝ pre * post
        # weights: (out, in), pre: (batch, in), post: (batch, out)
        
        batch_size = pre_spikes.size(0)
        
        # Correlation-based update
        delta_w = torch.einsum('bo,bi->oi', post_spikes, pre_spikes) / batch_size
        
        # Weight decay for stability
        delta_w = delta_w - self.weight_decay * weights
        
        # Scale by learning rate
        delta_w = self.lr * delta_w
        
        # Clip update magnitude
        delta_w = torch.clamp(delta_w, -0.1, 0.1)
        
        return delta_w
    
    def apply_update(self, weights: torch.Tensor, delta_w: torch.Tensor) -> torch.Tensor:
        """Apply update with normalization."""
        new_weights = weights + delta_w
        
        # Normalize weights per output neuron
        norm = new_weights.norm(dim=1, keepdim=True)
        new_weights = new_weights / (norm + 1e-6) * weights.norm(dim=1, keepdim=True).mean()
        
        return new_weights


def evaluate_stable_stdp():
    """Evaluate stable STDP adaptation."""
    config = load_config('./config/config.yaml')
    set_seed(config['experiment']['seed'])
    device = torch.device(config['experiment']['device'])
    
    # Load data
    data_dir = './data/mitdb'
    all_records = config['data']['train_records'] + config['data']['test_records']
    existing_records = [r for r in all_records if os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    
    encoder = GradedDeltaEncoder(num_levels=config['encoding']['num_levels'], threshold_factor=config['encoding']['threshold_factor'])
    dataset = MITBIHDataset(data_dir, existing_records, window_size=config['data']['window_size'], encoder=encoder)
    
    # Load base model
    base_model = TemporalCSNN(input_size=config['data']['window_size']).to(device)
    model_path = './models/temporal_csnn_v2_fold0_best.pth'
    if os.path.exists(model_path):
        base_model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded base model")
    
    base_model.eval()
    
    # STDP learner
    stdp = StableSTDPLearner(lr=0.0001, weight_decay=0.001)
    
    print("\n" + "="*60)
    print("Stable STDP Personalization")
    print("="*60)
    
    results = []
    
    for record in existing_records[:4]:
        print(f"\n--- Record: {record} ---")
        
        record_mask = dataset.record_ids == record
        record_indices = np.where(record_mask)[0]
        
        if len(record_indices) < 20:
            continue
        
        np.random.shuffle(record_indices)
        adapt_indices = record_indices[:len(record_indices)//2]
        test_indices = record_indices[len(record_indices)//2:]
        
        adapt_subset = Subset(dataset, adapt_indices)
        test_subset = Subset(dataset, test_indices)
        
        adapt_loader = DataLoader(adapt_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32)
        
        # Store original weights
        original_weights = base_model.out.weight.data.clone()
        original_bias = base_model.out.bias.data.clone() if base_model.out.bias is not None else None
        
        # Evaluate before adaptation
        def evaluate_model():
            preds, targets, probs = [], [], []
            with torch.no_grad():
                for data, tgt in test_loader:
                    data = data.to(device)
                    _, output = base_model(data)
                    prob = torch.softmax(output, dim=1)
                    pred = torch.argmax(output, dim=1)
                    preds.extend(pred.cpu().numpy())
                    targets.extend(tgt.numpy())
                    probs.extend(prob[:, 1].cpu().numpy())
            return MetricsCalculator.calculate_all(np.array(targets), np.array(preds), np.array(probs))
        
        metrics_before = evaluate_model()
        
        # STDP adaptation (few-shot)
        num_adapt_batches = min(5, len(adapt_loader))  # Only use few batches
        
        for i, (data, _) in enumerate(adapt_loader):
            if i >= num_adapt_batches:
                break
            
            data = data.to(device)
            
            with torch.no_grad():
                # Get pre-synaptic spikes (from FC layer)
                x_in = data.unsqueeze(1)
                feat = torch.relu(base_model.bn1(base_model.conv1(x_in)))
                feat = torch.relu(base_model.bn2(base_model.conv2(feat)))
                feat_flat = feat.flatten(1)
                
                mem1 = base_model.lif1.init_leaky()
                mem2 = base_model.lif2.init_leaky()
                mem_fc = base_model.lif_fc.init_leaky()
                
                for t in range(base_model.num_steps):
                    spk1, mem1 = base_model.lif1(feat_flat, mem1)
                    spk2, mem2 = base_model.lif2(spk1, mem2)
                    cur_fc = base_model.fc(spk2)
                    spk_fc, mem_fc = base_model.lif_fc(cur_fc, mem_fc)
                
                pre_spikes = spk_fc  # (batch, 64)
                
                # Get post-synaptic activity
                output = base_model.out(pre_spikes)
                post_spikes = torch.sigmoid(output)  # (batch, 2)
                
                # Compute and apply STDP update
                delta_w = stdp.compute_update(pre_spikes, post_spikes, base_model.out.weight.data)
                base_model.out.weight.data = stdp.apply_update(base_model.out.weight.data, delta_w)
        
        # Evaluate after adaptation
        metrics_after = evaluate_model()
        
        print(f"  Before: Acc={metrics_before['accuracy']:.4f}, F1={metrics_before['macro_f1']:.4f}")
        print(f"  After:  Acc={metrics_after['accuracy']:.4f}, F1={metrics_after['macro_f1']:.4f}")
        
        # Restore original weights
        base_model.out.weight.data = original_weights
        if original_bias is not None:
            base_model.out.bias.data = original_bias
        
        results.append({
            'record': record,
            'before': metrics_before,
            'after': metrics_after,
            'improvement': {
                'accuracy': metrics_after['accuracy'] - metrics_before['accuracy'],
                'macro_f1': metrics_after['macro_f1'] - metrics_before['macro_f1']
            }
        })
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if results:
        avg_acc_imp = np.mean([r['improvement']['accuracy'] for r in results])
        avg_f1_imp = np.mean([r['improvement']['macro_f1'] for r in results])
        
        print(f"\nAverage Change:")
        print(f"  Accuracy: {avg_acc_imp:+.4f}")
        print(f"  Macro F1: {avg_f1_imp:+.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        records = [r['record'] for r in results]
        acc_before = [r['before']['accuracy'] for r in results]
        acc_after = [r['after']['accuracy'] for r in results]
        f1_before = [r['before']['macro_f1'] for r in results]
        f1_after = [r['after']['macro_f1'] for r in results]
        
        x = np.arange(len(records))
        width = 0.35
        
        axes[0].bar(x - width/2, acc_before, width, label='Before STDP', alpha=0.7, color='blue')
        axes[0].bar(x + width/2, acc_after, width, label='After STDP', alpha=0.7, color='orange')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy: Before vs After STDP')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'Rec {r}' for r in records])
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        
        axes[1].bar(x - width/2, f1_before, width, label='Before STDP', alpha=0.7, color='blue')
        axes[1].bar(x + width/2, f1_after, width, label='After STDP', alpha=0.7, color='orange')
        axes[1].set_ylabel('Macro F1')
        axes[1].set_title('Macro F1: Before vs After STDP')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Rec {r}' for r in records])
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('./results/stdp_stable.png', dpi=150)
        print("\nSaved to results/stdp_stable.png")
    
    return results


if __name__ == "__main__":
    evaluate_stable_stdp()
