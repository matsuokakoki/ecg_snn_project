"""
MIT-BIH Arrhythmia Database loader with patient-split and stratified k-fold.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import wfdb

class GradedDeltaEncoder:
    """
    Graded Delta Modulation encoder for ECG signals.
    Converts continuous ECG to multi-valued spike trains.
    """
    
    def __init__(self, num_levels: int = 3, threshold_factor: float = 0.5):
        self.num_levels = num_levels
        self.threshold_factor = threshold_factor
    
    def encode(self, signal: np.ndarray) -> np.ndarray:
        """
        Encode ECG signal using graded delta modulation.
        
        Args:
            signal: 1D ECG signal array
        
        Returns:
            Spike train with values in [-num_levels, num_levels]
        """
        # Dynamic threshold based on signal statistics
        threshold_base = np.std(signal) * self.threshold_factor
        
        spikes = np.zeros_like(signal)
        prev_val = signal[0]
        
        for i in range(1, len(signal)):
            diff = signal[i] - prev_val
            
            # Check each level from highest to lowest
            for level in range(self.num_levels, 0, -1):
                thresh = threshold_base * level
                if diff >= thresh:
                    spikes[i] = level
                    prev_val = signal[i]
                    break
                elif diff <= -thresh:
                    spikes[i] = -level
                    prev_val = signal[i]
                    break
        
        return spikes
    
    def compute_isi(self, spikes: np.ndarray) -> np.ndarray:
        """
        Compute Inter-Spike Intervals.
        
        Args:
            spikes: Spike train
        
        Returns:
            Array of ISI values
        """
        spike_times = np.where(spikes != 0)[0]
        if len(spike_times) < 2:
            return np.array([])
        return np.diff(spike_times)


class MITBIHDataset(Dataset):
    """
    MIT-BIH Arrhythmia Database Dataset with patient-split support.
    """
    
    # Normal beat symbols
    NORMAL_SYMBOLS = ['N', 'L', 'R', 'e', 'j']
    
    def __init__(self, data_dir: str, records: list, window_size: int = 360,
                 encoder: GradedDeltaEncoder = None, lead: int = 0):
        """
        Args:
            data_dir: Directory containing MIT-BIH data
            records: List of record names to load
            window_size: Window size around each beat
            encoder: Spike encoder instance
            lead: ECG lead index (0 for MLII)
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.encoder = encoder or GradedDeltaEncoder()
        self.lead = lead
        
        self.data = []
        self.labels = []
        self.record_ids = []  # Track which record each sample came from
        self.spike_data = []
        
        self._load_records(records)
    
    def _load_records(self, records: list):
        """Load and process records."""
        for record_name in records:
            try:
                record_path = os.path.join(self.data_dir, record_name)
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                
                signal = record.p_signal[:, self.lead]
                
                # Extract beats
                for sample, symbol in zip(annotation.sample, annotation.symbol):
                    # Skip if window would exceed signal bounds
                    start = sample - self.window_size // 2
                    end = sample + self.window_size // 2
                    if start < 0 or end > len(signal):
                        continue
                    
                    # Binary label: Normal (0) vs Abnormal (1)
                    label = 0 if symbol in self.NORMAL_SYMBOLS else 1
                    
                    # Extract segment
                    segment = signal[start:end]
                    
                    # Encode to spikes
                    spikes = self.encoder.encode(segment)
                    
                    self.data.append(segment)
                    self.spike_data.append(spikes)
                    self.labels.append(label)
                    self.record_ids.append(record_name)
                    
            except Exception as e:
                print(f"Warning: Could not load record {record_name}: {e}")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.spike_data = np.array(self.spike_data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.record_ids = np.array(self.record_ids)
        
        print(f"Loaded {len(self.labels)} beats from {len(set(self.record_ids))} records")
        print(f"Label distribution: Normal={np.sum(self.labels==0)}, Abnormal={np.sum(self.labels==1)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.spike_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
    
    def get_raw_signal(self, idx):
        """Get raw ECG signal for visualization."""
        return self.data[idx]
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data."""
        class_counts = np.bincount(self.labels)
        weights = 1.0 / class_counts
        return torch.tensor(weights, dtype=torch.float32)


class StratifiedPatientKFold:
    """
    Stratified K-Fold cross-validation with patient-level splitting.
    Ensures no patient appears in both train and test sets.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, dataset: MITBIHDataset):
        """
        Generate train/test indices for each fold.
        
        Args:
            dataset: MITBIHDataset instance
        
        Yields:
            train_indices, test_indices for each fold
        """
        # Get unique patients and their dominant label
        unique_records = np.unique(dataset.record_ids)
        record_labels = []
        
        for record in unique_records:
            mask = dataset.record_ids == record
            # Use majority label for stratification
            record_label = np.bincount(dataset.labels[mask]).argmax()
            record_labels.append(record_label)
        
        record_labels = np.array(record_labels)
        
        # Stratified split at patient level
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, 
                             random_state=self.random_state)
        
        for train_records_idx, test_records_idx in skf.split(unique_records, record_labels):
            train_records = unique_records[train_records_idx]
            test_records = unique_records[test_records_idx]
            
            # Get sample indices for each set
            train_mask = np.isin(dataset.record_ids, train_records)
            test_mask = np.isin(dataset.record_ids, test_records)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices


def download_mitdb(data_dir: str, records: list = None):
    """Download MIT-BIH database records."""
    os.makedirs(data_dir, exist_ok=True)
    
    if records is None:
        # Default: download all records
        records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                   '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
                   '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
                   '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
                   '222', '223', '228', '230', '231', '232', '233', '234']
    
    print(f"Downloading {len(records)} records to {data_dir}...")
    
    for record in records:
        try:
            wfdb.dl_database('mitdb', data_dir, records=[record])
            print(f"  Downloaded: {record}")
        except Exception as e:
            print(f"  Failed to download {record}: {e}")
    
    print("Download complete.")


if __name__ == "__main__":
    # Test the data loader
    data_dir = "/home/ubuntu/ecg_snn_project/data/mitdb"
    
    # Download a few records for testing
    download_mitdb(data_dir, records=['100', '101', '103', '105', '106', '108'])
    
    # Create dataset
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    dataset = MITBIHDataset(data_dir, ['100', '101'], encoder=encoder)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0][0].shape}")
    print(f"Class weights: {dataset.get_class_weights()}")
