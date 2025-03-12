import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_team_performance(df, team_name):
    """
    Plot team performance metrics
    
    Args:
        df (DataFrame): Preprocessed team data
        team_name (str): Name of the team for plot titles
    """

    plt.figure(figsize=(15, 10))
    
    # Plot win/loss trend
    plt.subplot(2, 2, 1)
    rolling_wins = df['Win'].rolling(window=10).mean()
    plt.plot(df['Date'], rolling_wins)
    plt.title(f'{team_name} 10-Game Rolling Win %')
    plt.ylim(0, 1)
    plt.grid(True)
    
    # Plot points scored and allowed
    plt.subplot(2, 2, 2)
    plt.plot(df['Date'], df['Team_pts'], label='Points Scored', color='blue')
    plt.plot(df['Date'], df['Opp_pts'], label='Points Allowed', color='red')
    plt.title(f'{team_name} Points Scored vs Allowed')
    plt.legend()
    plt.grid(True)
    
    # Plot field goals made
    plt.subplot(2, 2, 3)
    plt.plot(df['Date'], df['Team_FG'], color='blue')
    plt.title(f'{team_name} Field Goals Made')
    plt.legend()
    plt.grid(True)
    
    # Plot efficiency metrics
    plt.subplot(2, 2, 4)
    plt.plot(df['Date'], df['eFG%'], label='eFG%', color='purple')
    plt.plot(df['Date'], df['AST_TO_Ratio'], label='AST/TO Ratio', color='brown')
    plt.title(f'{team_name} Efficiency Metrics')
    plt.legend()
    plt.grid(True)
    
    # Plot
    plt.tight_layout()
    plt.savefig(f"{team_name.lower()}_performance.png")
    plt.show()


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training and validation loss and accuracy

    Args:
        train_losses (list): Training loss values for each epoch
        val_losses (list): Validation loss values for each epoch
        train_accuracies (list): Training accuracy values for each epoch
        val_accuracies (list): Validation accuracy values for each epoch
    """
    
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probas):
    """
    Plot the ROC curve and calculate AUC score
    
    Args:
        y_true (list or array): True labels
        y_probas (list or array): Predicted labels 
    
    Returns:
        roc_auc (float): AUC score
    """

    # Convert inputs to numpy arrays
    y_true = np.array(y_true).flatten()
    y_probas = np.array(y_probas).flatten()
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_probas)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))

    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', 
             label='Random')
    
    # Add labels and formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add threshold annotations
    threshold_indices = [10, 25, 50, 75, 90]
    for i in threshold_indices:
        if i < len(thresholds):
            idx = i
            plt.annotate(f'Threshold: {thresholds[idx]:.2f}', 
                        xy=(fpr[idx], tpr[idx]), 
                        xytext=(fpr[idx]+0.05, tpr[idx]-0.05),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    
    # Plot
    plt.tight_layout()
    plt.show()
    
    return roc_auc