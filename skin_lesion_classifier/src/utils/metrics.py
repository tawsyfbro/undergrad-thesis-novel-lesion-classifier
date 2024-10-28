# src/utils/metrics.py
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:

    return {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred)
    }
