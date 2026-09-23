"""Train with disjoint train, validation and test subjects.

This pipeline has not been used to produce any results committed to this repo.
Run from the repository root after obtaining the MIT-BIH data. It intentionally
does not reuse historical checkpoints or metrics.
"""

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_loader import GradedDeltaEncoder, MITBIHDataset
from evaluation_protocol import split_training_records, validate_subject_disjoint
from snn_model_v2 import TemporalCSNN
from train_final_optimized import (
    AsymmetricLoss,
    AugmentedDataset,
    ImprovedCNN,
    evaluate_with_threshold,
    find_optimal_threshold_aggressive,
    train_epoch,
)
from utils import load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]


def require_records(data_dir, records):
    missing = [r for r in records if not all((data_dir / f"{r}.{ext}").exists() for ext in ("hea", "dat", "atr"))]
    if missing:
        raise FileNotFoundError(f"MIT-BIH records missing from {data_dir}: {missing}")


def require_two_classes(dataset, name):
    if set(np.unique(dataset.labels)) != {0, 1}:
        raise ValueError(f"{name} must contain both classes")


def run(model_type, seed=42):
    config = load_config(str(ROOT / "config" / "config.yaml"))
    train_pool = [str(r) for r in config["data"]["train_records"]]
    # The historical config places 201 in train and 202 in test, although
    # PhysioNet identifies them as recordings from one subject. Exclude 202
    # from the corrected test protocol; leave the original config unchanged
    # so the historical experiment remains interpretable.
    test_records = [str(r) for r in config["data"]["test_records"] if str(r) != "202"]
    train_records, val_records = split_training_records(train_pool, seed=seed)
    validate_subject_disjoint(train_records, val_records, test_records)

    data_dir = ROOT / "data" / "mitdb"
    require_records(data_dir, train_records + val_records + test_records)
    encoder = GradedDeltaEncoder(num_levels=3, threshold_factor=0.5)
    train_pool_data = MITBIHDataset(str(data_dir), train_pool, window_size=360, encoder=encoder)
    test_data = MITBIHDataset(str(data_dir), test_records, window_size=360, encoder=encoder)
    train_idx = np.flatnonzero(np.isin(train_pool_data.record_ids, train_records))
    val_idx = np.flatnonzero(np.isin(train_pool_data.record_ids, val_records))
    if not len(train_idx) or not len(val_idx) or not len(test_data):
        raise ValueError("A partition is empty")
    require_two_classes(SubsetLabels(train_pool_data.labels[train_idx]), "train")
    require_two_classes(SubsetLabels(train_pool_data.labels[val_idx]), "validation")
    require_two_classes(test_data, "test")

    set_seed(seed)
    device = torch.device(config["experiment"]["device"])
    train_loader = DataLoader(AugmentedDataset(train_pool_data, train_idx, oversample_ratio=8), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(train_pool_data, val_idx), batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    model = (TemporalCSNN(input_size=360) if model_type == "snn" else ImprovedCNN(input_size=360)).to(device)
    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=2, clip=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_sensitivity = float("-inf")
    best_state = None
    patience = 0
    for epoch in range(100):
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=0.2)
        scheduler.step()
        if val_metrics["sensitivity"] > best_sensitivity:
            best_sensitivity = val_metrics["sensitivity"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 20:
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    _, val_targets, val_probs = evaluate_with_threshold(model, val_loader, device)
    threshold = find_optimal_threshold_aggressive(val_targets, val_probs, target_sensitivity=0.80)
    test_metrics, _, _ = evaluate_with_threshold(model, test_loader, device, threshold=threshold)
    print(f"{model_type}: validation-selected threshold={threshold:.6f}; independent test={test_metrics}")
    return test_metrics


class SubsetLabels:
    def __init__(self, labels):
        self.labels = labels


if __name__ == "__main__":
    run("snn")
    run("cnn")
