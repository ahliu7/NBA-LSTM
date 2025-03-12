import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics for model evaluation
    
    Args:
        y_true (list): True labels 
        y_pred (list): Predicted labels
        
    Returns:
        precision (float): Precision metric
        recall (float): Recall metric
        f1 (float): F1 score 
        confusion (array): 2x2 confusion matrix 
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, confusion