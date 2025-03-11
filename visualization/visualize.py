import matplotlib.pyplot as plt

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