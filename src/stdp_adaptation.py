"""
STDP (Spike-Timing-Dependent Plasticity) for on-chip personalization.
Implements Δt-based STDP learning rule for adapting to individual ECG patterns.
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

from utils import set_seed, load_config, MetricsCalculator, print_metrics
from data_loader import MITBIHDataset, GradedDeltaEncoder, StratifiedPatientKFold
from snn_model_v2 import TemporalCSNN


class STDPLearner:
    """
    STDP learning rule implementation.
    
    Δw = A+ * exp(-Δt/τ+) if Δt > 0 (pre before post)
    Δw = -A- * exp(Δt/τ-) if Δt < 0 (post before pre)
    
    where Δt = t_post - t_pre
    """
    
    def __init__(self, tau_pre: float = 20.0, tau_post: float = 20.0,
                 lr_plus: float = 0.01, lr_minus: float = 0.012,
                 weight_min: float = -1.0, weight_max: float = 1.0):
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.weight_min = weight_min
        self.weight_max = weight_max
        
        # Trace variables for online learning
        self.pre_trace = None
        self.post_trace = None
    
    def compute_weight_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                             weights: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Compute weight update based on spike timing.
        
        Args:
            pre_spikes: Pre-synaptic spikes (batch, pre_neurons)
            post_spikes: Post-synaptic spikes (batch, post_neurons)
            weights: Current weights (out_neurons, in_neurons) - transposed format
            dt: Time step duration
        
        Returns:
            Weight update tensor matching weights shape
        """
        batch_size = pre_spikes.size(0)
        
        # Initialize traces if needed
        if self.pre_trace is None or self.pre_trace.size() != pre_spikes.size():
            self.pre_trace = torch.zeros_like(pre_spikes)
        if self.post_trace is None or self.post_trace.size() != post_spikes.size():
            self.post_trace = torch.zeros_like(post_spikes)
        
        # Decay traces
        self.pre_trace = self.pre_trace * np.exp(-dt / self.tau_pre)
        self.post_trace = self.post_trace * np.exp(-dt / self.tau_post)
        
        # Update traces with new spikes
        self.pre_trace = self.pre_trace + pre_spikes
        self.post_trace = self.post_trace + post_spikes
        
        # Compute weight updates
        # LTP: pre before post (pre_trace * post_spikes)
        # LTD: post before pre (post_trace * pre_spikes)
        
        # Outer product for weight update
        # weights shape: (out_neurons, in_neurons) = (2, 64)
        # pre_spikes: (batch, 64), post_spikes: (batch, 2)
        # delta_w should be (2, 64)
        ltp = torch.einsum('bq,bp->qp', post_spikes, self.pre_trace)  # LTP: (out, in)
        ltd = torch.einsum('bq,bp->qp', self.post_trace, pre_spikes)  # LTD: (out, in)
        
        # Average over batch
        delta_w = (self.lr_plus * ltp - self.lr_minus * ltd) / batch_size
        
        return delta_w
    
    def apply_update(self, weights: torch.Tensor, delta_w: torch.Tensor) -> torch.Tensor:
        """Apply weight update with bounds."""
        new_weights = weights + delta_w
        return torch.clamp(new_weights, self.weight_min, self.weight_max)
    
    def reset_traces(self):
        """Reset trace variables."""
        self.pre_trace = None
        self.post_trace = None


class STDPAdaptiveModel(nn.Module):
    """
    SNN model with STDP-adaptable final layer.
    The feature extraction layers are frozen, only the final layer adapts via STDP.
    """
    
    def __init__(self, base_model: TemporalCSNN, stdp_learner: STDPLearner):
        super().__init__()
        self.base_model = base_model
        self.stdp_learner = stdp_learner
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create adaptable final layer
        self.adaptive_weights = nn.Parameter(
            self.base_model.out.weight.data.clone(),
            requires_grad=False
        )
        self.adaptive_bias = nn.Parameter(
            self.base_model.out.bias.data.clone() if self.base_model.out.bias is not None else torch.zeros(2),
            requires_grad=False
        )
        
        # Store spike history for STDP
        self.pre_spikes_history = []
        self.post_spikes_history = []
    
    def forward(self, x, adapt: bool = False):
        """Forward pass with optional STDP adaptation."""
        # Get features from base model (up to last layer)
        self.base_model.eval()
        
        batch_size = x.size(0)
        device = x.device
        
        # Add channel dimension
        x_in = x.unsqueeze(1)
        
        # Feature extraction
        feat = torch.relu(self.base_model.bn1(self.base_model.conv1(x_in)))
        feat = torch.relu(self.base_model.bn2(self.base_model.conv2(feat)))
        feat_flat = feat.flatten(1)
        
        # Temporal processing
        mem1 = self.base_model.lif1.init_leaky()
        mem2 = self.base_model.lif2.init_leaky()
        mem_fc = self.base_model.lif_fc.init_leaky()
        
        spk_rec = []
        
        for t in range(self.base_model.num_steps):
            spk1, mem1 = self.base_model.lif1(feat_flat, mem1)
            spk2, mem2 = self.base_model.lif2(spk1, mem2)
            cur_fc = self.base_model.fc(spk2)
            spk_fc, mem_fc = self.base_model.lif_fc(cur_fc, mem_fc)
            spk_rec.append(spk_fc)
            
            # Store spikes for STDP
            if adapt:
                self.pre_spikes_history.append(spk_fc.detach())
        
        # Aggregate temporal output
        spk_out = torch.stack(spk_rec, dim=0).sum(dim=0)
        
        # Adaptive output layer
        out = torch.matmul(spk_out, self.adaptive_weights.T) + self.adaptive_bias
        
        # Post-synaptic spikes (use output as proxy)
        if adapt:
            post_spikes = (out > 0).float()
            self.post_spikes_history.append(post_spikes.detach())
        
        return spk_out, out
    
    def adapt_stdp(self, num_steps: int = 1):
        """Apply STDP adaptation based on stored spike history."""
        if not self.pre_spikes_history or not self.post_spikes_history:
            return
        
        # Average over time steps
        pre_spikes = torch.stack(self.pre_spikes_history, dim=0).mean(dim=0)
        post_spikes = torch.stack(self.post_spikes_history, dim=0).mean(dim=0)
        
        # Compute weight update
        delta_w = self.stdp_learner.compute_weight_update(
            pre_spikes, post_spikes, self.adaptive_weights.data
        )
        
        # Apply update
        self.adaptive_weights.data = self.stdp_learner.apply_update(
            self.adaptive_weights.data, delta_w
        )
        
        # Clear history
        self.pre_spikes_history = []
        self.post_spikes_history = []
    
    def reset_adaptation(self):
        """Reset to original weights."""
        self.adaptive_weights.data = self.base_model.out.weight.data.clone()
        if self.base_model.out.bias is not None:
            self.adaptive_bias.data = self.base_model.out.bias.data.clone()
        self.stdp_learner.reset_traces()
        self.pre_spikes_history = []
        self.post_spikes_history = []


def evaluate_stdp_adaptation():
    """Evaluate STDP-based personalization."""
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
    
    # Create STDP learner
    stdp_learner = STDPLearner(
        tau_pre=config['stdp']['tau_pre'],
        tau_post=config['stdp']['tau_post'],
        lr_plus=config['stdp']['lr_plus'],
        lr_minus=config['stdp']['lr_minus'],
        weight_min=config['stdp']['weight_min'],
        weight_max=config['stdp']['weight_max']
    )
    
    # Create adaptive model
    adaptive_model = STDPAdaptiveModel(base_model, stdp_learner).to(device)
    
    # Simulate personalization for different "patients" (records)
    print("\n" + "="*60)
    print("STDP Personalization Simulation")
    print("="*60)
    
    results = []
    
    for record in existing_records[:4]:  # Test on first 4 records
        print(f"\n--- Patient/Record: {record} ---")
        
        # Get data for this record
        record_mask = dataset.record_ids == record
        record_indices = np.where(record_mask)[0]
        
        if len(record_indices) < 20:
            print(f"  Skipping: insufficient data ({len(record_indices)} samples)")
            continue
        
        # Split into adaptation and test sets
        np.random.shuffle(record_indices)
        adapt_indices = record_indices[:len(record_indices)//2]
        test_indices = record_indices[len(record_indices)//2:]
        
        adapt_subset = Subset(dataset, adapt_indices)
        test_subset = Subset(dataset, test_indices)
        
        adapt_loader = DataLoader(adapt_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32)
        
        # Evaluate before adaptation
        adaptive_model.reset_adaptation()
        
        preds_before, targets_before, probs_before = [], [], []
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                _, output = adaptive_model(data, adapt=False)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                preds_before.extend(preds.cpu().numpy())
                targets_before.extend(targets.numpy())
                probs_before.extend(probs[:, 1].cpu().numpy())
        
        metrics_before = MetricsCalculator.calculate_all(
            np.array(targets_before), np.array(preds_before), np.array(probs_before)
        )
        
        # STDP adaptation
        print(f"  Adapting with {len(adapt_indices)} samples...")
        
        for epoch in range(config['stdp']['adaptation_steps'] // len(adapt_loader) + 1):
            for data, targets in adapt_loader:
                data = data.to(device)
                _, _ = adaptive_model(data, adapt=True)
                adaptive_model.adapt_stdp()
        
        # Evaluate after adaptation
        preds_after, targets_after, probs_after = [], [], []
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                _, output = adaptive_model(data, adapt=False)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                preds_after.extend(preds.cpu().numpy())
                targets_after.extend(targets.numpy())
                probs_after.extend(probs[:, 1].cpu().numpy())
        
        metrics_after = MetricsCalculator.calculate_all(
            np.array(targets_after), np.array(preds_after), np.array(probs_after)
        )
        
        print(f"  Before STDP: Acc={metrics_before['accuracy']:.4f}, F1={metrics_before['macro_f1']:.4f}")
        print(f"  After STDP:  Acc={metrics_after['accuracy']:.4f}, F1={metrics_after['macro_f1']:.4f}")
        
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
    print("STDP Adaptation Summary")
    print("="*60)
    
    if results:
        avg_acc_improvement = np.mean([r['improvement']['accuracy'] for r in results])
        avg_f1_improvement = np.mean([r['improvement']['macro_f1'] for r in results])
        
        print(f"\nAverage Improvement:")
        print(f"  Accuracy: {avg_acc_improvement:+.4f}")
        print(f"  Macro F1: {avg_f1_improvement:+.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        records = [r['record'] for r in results]
        acc_before = [r['before']['accuracy'] for r in results]
        acc_after = [r['after']['accuracy'] for r in results]
        f1_before = [r['before']['macro_f1'] for r in results]
        f1_after = [r['after']['macro_f1'] for r in results]
        
        x = np.arange(len(records))
        width = 0.35
        
        axes[0].bar(x - width/2, acc_before, width, label='Before STDP', alpha=0.7)
        axes[0].bar(x + width/2, acc_after, width, label='After STDP', alpha=0.7)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Before/After STDP Adaptation')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'Record {r}' for r in records])
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        
        axes[1].bar(x - width/2, f1_before, width, label='Before STDP', alpha=0.7)
        axes[1].bar(x + width/2, f1_after, width, label='After STDP', alpha=0.7)
        axes[1].set_ylabel('Macro F1')
        axes[1].set_title('Macro F1 Before/After STDP Adaptation')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Record {r}' for r in records])
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('./results/stdp_adaptation.png', dpi=150)
        print("\nSaved plot to results/stdp_adaptation.png")
    
    return results


if __name__ == "__main__":
    evaluate_stdp_adaptation()
