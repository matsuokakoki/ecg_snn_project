"""
Utility functions for reproducibility, logging, and evaluation metrics.
"""
import os
import json
import yaml
import random
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: dict, save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

class ExperimentLogger:
    """Logger for experiment tracking."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        self.log_path = os.path.join(log_dir, f"{self.experiment_id}.json")
        self.logs = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "training_history": [],
            "evaluation_results": {},
            "config": {}
        }
    
    def log_config(self, config: dict):
        self.logs["config"] = config
    
    def log_epoch(self, epoch: int, train_loss: float, val_metrics: dict = None):
        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_metrics:
            entry.update(val_metrics)
        self.logs["training_history"].append(entry)
    
    def log_evaluation(self, fold: int, metrics: dict):
        self.logs["evaluation_results"][f"fold_{fold}"] = metrics
    
    def log_final_results(self, results: dict):
        self.logs["final_results"] = results
    
    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to {self.log_path}")

class MetricsCalculator:
    """Calculate evaluation metrics for ECG classification."""
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities for positive class (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_f1"] = f1_score(y_true, y_pred, average='macro')
        metrics["weighted_f1"] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for positive class
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # PR-AUC (if probabilities available)
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = auc(recall, precision)
        
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics
    
    @staticmethod
    def aggregate_fold_results(fold_results: list) -> dict:
        """
        Aggregate results from k-fold cross-validation.
        
        Args:
            fold_results: List of metric dictionaries from each fold
        
        Returns:
            Dictionary with mean ± std for each metric
        """
        aggregated = {}
        metric_names = ["accuracy", "macro_f1", "weighted_f1", "sensitivity", "specificity", "pr_auc"]
        
        for metric in metric_names:
            values = [r.get(metric, np.nan) for r in fold_results if metric in r]
            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                # 95% CI (assuming normal distribution)
                aggregated[f"{metric}_ci95"] = 1.96 * np.std(values) / np.sqrt(len(values))
        
        return aggregated

def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print('='*50)
    for key, value in metrics.items():
        if key != "confusion_matrix":
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    print('='*50)
