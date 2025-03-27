from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinarySpecificity, BinaryF1Score, BinaryConfusionMatrix
)
from functools import reduce
import torch

import pandas as pd



def accum_tensor(t1, t2, func, idx):
    if len(t1.shape) == 0:
        return func(t1.item(), t2.item())
    cur_t1 = t1[idx]
    cur_t2 = t2[idx]
    res = accum_tensor(cur_t1, cur_t2, func, 0)
    if idx == t1.shape[0] - 1:
        return res 
    return res + accum_tensor(t1, t2, func, idx + 1)



def compute_metrics(preds, labels, device='cuda'):
    """
    Compute accuracy, precision, recall (sensitivity), specificity, NPV, and F1 score using TorchMetrics.
    
    Args:
        preds (torch.Tensor): Predicted probabilities (or logits).
        labels (torch.Tensor): Ground truth binary labels (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing all relevant metrics.
    """

    # Initialize metrics
    accuracy_metric = BinaryAccuracy().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)  # Sensitivity
    specificity_metric = BinarySpecificity().to(device)
    f1_metric = BinaryF1Score().to(device)
    confusion_matrix_metric = BinaryConfusionMatrix().to(device)

    # Compute standard metrics
    accuracy = accuracy_metric(preds, labels)
    precision = precision_metric(preds, labels)
    recall = recall_metric(preds, labels)
    specificity = specificity_metric(preds, labels)
    f1_score = f1_metric(preds, labels)

    # Compute Confusion Matrix (to get TN & FN)
    confusion_matrix = confusion_matrix_metric(preds, labels)
    tn, fp, fn, tp = confusion_matrix.flatten()  # Extract values

    # Compute NPV (Avoid division by zero)
    npv = tn / (tn + fn) if (tn + fn) > 0 else torch.tensor(0.0, device=device)

    # Return all metrics as a dictionary
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),  # Sensitivity
        "specificity": specificity.item(),
        "f1_score": f1_score.item(),
        "npv": npv.item()  # Negative Predictive Value
    }
    

if __name__ == "__main__":
    thresholds = {
        'val_acc': 0.4,
        'val_auc_roc': 0.5,
        'val_rec': 0.5
    }
    filtered = filter_val_rows('your_file.csv', thresholds)
    print(filtered)