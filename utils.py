from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinarySpecificity, BinaryF1Score
from functools import reduce

def accum_tensor(t1, t2, func, idx):
    if len(t1.shape) == 0:
        return func(t1.item(), t2.item())
    cur_t1 = t1[idx]
    cur_t2 = t2[idx]
    res = accum_tensor(cur_t1, cur_t2, func, 0)
    if idx == t1.shape[0] - 1:
        return res 
    return res + accum_tensor(t1, t2, func, idx + 1)

def compute_metrics(preds, labels, threshold=0.5, device='cuda'):
    """
    Compute accuracy, precision, recall (sensitivity), specificity, and F1 score using TorchMetrics.
    
    Args:
        preds (torch.Tensor): Predicted probabilities (or logits).
        labels (torch.Tensor): Ground truth binary labels (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.
    
    Returns:
        dict: A dictionary containing all relevant metrics.
    """
    # Convert probabilities to binary predictions

    preds_binary = preds

    # Initialize metrics
    accuracy_metric = BinaryAccuracy().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    specificity_metric = BinarySpecificity().to(device)
    f1_metric = BinaryF1Score().to(device)

    # Compute metrics
    accuracy = accuracy_metric(preds_binary, labels)
    precision = precision_metric(preds_binary, labels)
    recall = recall_metric(preds_binary, labels)
    specificity = specificity_metric(preds_binary, labels)
    f1_score = f1_metric(preds_binary, labels)

    # Return all metrics as a dictionary
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),  # Sensitivity
        "specificity": specificity.item(),
        "f1_score": f1_score.item()
    }