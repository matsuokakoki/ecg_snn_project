"""
Improved Temporal 1D Convolutional Spiking Neural Network (1D-CSNN).
More stable training with proper temporal processing.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np

class TemporalCSNN(nn.Module):
    """
    1D Convolutional SNN with explicit temporal processing.
    Modified to return spike records for SOPs analysis.
    """
    
    def __init__(self, input_size: int = 360, num_steps: int = 20, beta: float = 0.85,
                 conv1_channels: int = 16, conv2_channels: int = 32, 
                 kernel_size: int = 7, stride: int = 2, num_classes: int = 2,
                 surrogate_slope: float = 25.0):
        super().__init__()
        
        self.input_size = input_size
        self.num_steps = num_steps
        self.beta = beta
        
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        
        # Non-spiking feature extraction layers
        self.conv1 = nn.Conv1d(1, conv1_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        
        # Calculate output size after convolutions
        conv1_out = (input_size + 2*(kernel_size//2) - kernel_size) // stride + 1
        conv2_out = (conv1_out + 2*(kernel_size//2) - kernel_size) // stride + 1
        self.fc_input_size = conv2_channels * conv2_out
        
        # Spiking layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.fc = nn.Linear(self.fc_input_size, 64)
        self.lif_fc = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.out = nn.Linear(64, num_classes)
        
        # For SOPs tracking
        self.track_spikes = True

    def forward(self, x):
        batch_size = x.size(0)
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Non-spiking feature extraction
        feat = torch.relu(self.bn1(self.conv1(x)))
        feat = torch.relu(self.bn2(self.conv2(feat)))
        feat_flat = feat.flatten(1)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem_fc = self.lif_fc.init_leaky()
        
        # Spike recording
        spike_records = {
            'conv1': [], 'conv2': [], 'conv3': [], # Dummy, not real spikes
            'fc1': [], 'fc2': []
        }
        
        # Temporal processing loop
        for t in range(self.num_steps):
            # Inject features as current into the first spiking layer
            spk1, mem1 = self.lif1(feat_flat, mem1)
            
            # FC layer
            cur_fc = self.fc(spk1)
            spk_fc, mem_fc = self.lif_fc(cur_fc, mem_fc)
            
            # Record spikes for this timestep
            if self.track_spikes:
                spike_records["fc1"].append(spk1)
                spike_records["fc2"].append(spk_fc)

        # Final output from the last membrane potential of the output layer
        out = self.out(mem_fc)
        
        if self.track_spikes:
            # Stack spikes over time
            for key in ["fc1", "fc2"]:
                spike_records[key] = torch.stack(spike_records[key], dim=2)
            
            # Add dummy tensors for conv layers to match analysis script structure
            spike_records["conv1"] = torch.zeros_like(spike_records["fc1"])
            spike_records["conv2"] = torch.zeros_like(spike_records["fc1"])
            spike_records["conv3"] = torch.zeros_like(spike_records["fc1"])

            return spike_records, out
        
        return out
