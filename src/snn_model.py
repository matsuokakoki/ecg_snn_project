"""
Temporal 1D Convolutional Spiking Neural Network (1D-CSNN).
Processes input through explicit time step loop with stateful membrane potentials.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np

class TemporalCSNN(nn.Module):
    """
    1D Convolutional SNN with explicit temporal processing.
    
    Key features:
    - Time step loop for temporal coding
    - Stateful membrane potentials across time
    - Spike counting for SOPs calculation
    """
    
    def __init__(self, input_size: int = 360, num_steps: int = 50, beta: float = 0.9,
                 conv1_channels: int = 8, conv2_channels: int = 16, 
                 kernel_size: int = 5, stride: int = 2, num_classes: int = 2,
                 surrogate_slope: float = 25.0):
        super().__init__()
        
        self.input_size = input_size
        self.num_steps = num_steps
        self.beta = beta
        
        # Surrogate gradient function
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        
        # Convolutional layers (applied once to full input)
        self.conv1 = nn.Conv1d(1, conv1_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, stride=stride)
        
        # Calculate output size after convolutions
        conv1_out = (input_size - kernel_size) // stride + 1
        conv2_out = (conv1_out - kernel_size) // stride + 1
        self.fc_input_size = conv2_channels * conv2_out
        
        # LIF neurons for temporal processing
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Fully connected output layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Store sizes for SOPs calculation
        self.conv1_out_size = conv1_out
        self.conv2_out_size = conv2_out
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        
        # For SOPs tracking
        self.spike_counts = {}
        self.track_spikes = False
    
    def forward(self, x):
        """
        Forward pass with explicit time step processing.
        
        Args:
            x: Input tensor of shape (batch, input_size) - graded spike encoded ECG
        
        Returns:
            spk_out: Output spikes accumulated over time
            mem_out: Final membrane potential (used for classification)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # Apply convolutions to get feature maps
        feat1 = self.conv1(x)  # (batch, conv1_channels, conv1_out)
        feat2 = self.conv2(feat1)  # (batch, conv2_channels, conv2_out)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        # Spike recording for SOPs
        if self.track_spikes:
            self.spike_counts = {
                'layer1': 0,
                'layer2': 0,
                'output': 0
            }
        
        # Accumulate output over time
        spk_out_sum = torch.zeros(batch_size, 2, device=device)
        
        # Temporal processing: iterate through time steps
        # Each time step processes a portion of the feature map
        feat1_time_size = feat1.size(2)
        feat2_time_size = feat2.size(2)
        
        for t in range(self.num_steps):
            # Get time-indexed features (with interpolation for smooth temporal coverage)
            t_ratio = t / max(self.num_steps - 1, 1)
            
            # Layer 1: Sample from conv1 features
            t1_idx = int(t_ratio * (feat1_time_size - 1))
            cur1 = feat1[:, :, t1_idx]  # (batch, conv1_channels)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            if self.track_spikes:
                self.spike_counts['layer1'] += spk1.sum().item()
            
            # Layer 2: Sample from conv2 features, modulated by layer1 spikes
            t2_idx = int(t_ratio * (feat2_time_size - 1))
            cur2 = feat2[:, :, t2_idx] * spk1.mean(dim=1, keepdim=True)  # Modulate by spike activity
            spk2, mem2 = self.lif2(cur2, mem2)
            
            if self.track_spikes:
                self.spike_counts['layer2'] += spk2.sum().item()
            
            # Output layer
            cur_out = self.fc(feat2.flatten(1)) * spk2.mean(dim=1, keepdim=True)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            if self.track_spikes:
                self.spike_counts['output'] += spk_out.sum().item()
            
            spk_out_sum += spk_out
        
        # Use final membrane potential for classification
        return spk_out_sum, mem_out
    
    def enable_spike_tracking(self):
        """Enable spike counting for SOPs calculation."""
        self.track_spikes = True
    
    def disable_spike_tracking(self):
        """Disable spike counting."""
        self.track_spikes = False
    
    def get_spike_counts(self):
        """Return spike counts from last forward pass."""
        return self.spike_counts
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_synapse_counts(self):
        """Get number of synapses (connections) per layer."""
        synapse_counts = {}
        
        # Conv1: in_channels * out_channels * kernel_size * output_size
        synapse_counts['conv1'] = (self.conv1.in_channels * self.conv1.out_channels * 
                                   self.conv1.kernel_size[0] * self.conv1_out_size)
        
        # Conv2
        synapse_counts['conv2'] = (self.conv2.in_channels * self.conv2.out_channels * 
                                   self.conv2.kernel_size[0] * self.conv2_out_size)
        
        # FC
        synapse_counts['fc'] = self.fc.in_features * self.fc.out_features
        
        synapse_counts['total'] = sum(synapse_counts.values())
        
        return synapse_counts
    
    def calculate_sops(self, spike_counts: dict = None):
        """
        Calculate Synaptic Operations (SOPs) based on spike counts.
        SOPs = sum over layers of (spike_count * fan_out)
        """
        if spike_counts is None:
            spike_counts = self.spike_counts
        
        synapse_counts = self.get_synapse_counts()
        
        sops = {
            'layer1': spike_counts.get('layer1', 0) * self.conv2.in_channels * self.conv2.kernel_size[0],
            'layer2': spike_counts.get('layer2', 0) * self.fc.in_features // self.conv2_channels,
            'output': spike_counts.get('output', 0) * 1  # Output layer has no fan-out
        }
        sops['total'] = sum(sops.values())
        
        return sops


class BaselineCNN(nn.Module):
    """
    Lightweight baseline CNN with similar parameter count to SNN.
    For fair comparison of accuracy and computational cost.
    """
    
    def __init__(self, input_size: int = 360, conv1_channels: int = 8, 
                 conv2_channels: int = 16, kernel_size: int = 5, 
                 stride: int = 2, num_classes: int = 2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, conv1_channels, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.relu2 = nn.ReLU()
        
        # Calculate output size
        conv1_out = (input_size - kernel_size) // stride + 1
        conv2_out = (conv1_out - kernel_size) // stride + 1
        self.fc_input_size = conv2_channels * conv2_out
        
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
        # Store for FLOPs calculation
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_size)
        
        Returns:
            Output logits
        """
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_flops(self):
        """
        Estimate FLOPs for one forward pass.
        """
        flops = 0
        
        # Conv1: 2 * K * Cin * Cout * Hout (multiply-add)
        flops += 2 * self.conv1.kernel_size[0] * self.conv1.in_channels * self.conv1.out_channels * self.conv1_out
        
        # Conv2
        flops += 2 * self.conv2.kernel_size[0] * self.conv2.in_channels * self.conv2.out_channels * self.conv2_out
        
        # FC
        flops += 2 * self.fc.in_features * self.fc.out_features
        
        return flops


if __name__ == "__main__":
    # Test models
    print("Testing Temporal CSNN...")
    snn_model = TemporalCSNN(input_size=360, num_steps=50)
    snn_model.enable_spike_tracking()
    
    x = torch.randn(4, 360)  # Batch of 4
    spk_out, mem_out = snn_model(x)
    
    print(f"SNN Parameters: {snn_model.count_parameters()}")
    print(f"SNN Synapse counts: {snn_model.get_synapse_counts()}")
    print(f"SNN Spike counts: {snn_model.get_spike_counts()}")
    print(f"SNN SOPs: {snn_model.calculate_sops()}")
    print(f"Output shape: spk={spk_out.shape}, mem={mem_out.shape}")
    
    print("\nTesting Baseline CNN...")
    cnn_model = BaselineCNN(input_size=360)
    out = cnn_model(x)
    
    print(f"CNN Parameters: {cnn_model.count_parameters()}")
    print(f"CNN FLOPs: {cnn_model.count_flops()}")
    print(f"Output shape: {out.shape}")
