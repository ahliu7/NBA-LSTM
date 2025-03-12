import torch
import torch.nn as nn
import torch.optim as optim
import os
import config
from datetime import datetime
from model.model import NBA_LSTM
from utils import calculate_metrics
from config import Constants

def train_model(train_loader, val_loader, input_size, device, epochs, learning_rate, patience):
    """
    Train the LSTM model

    Args:
        train_loader (DataLoader): PyTorch DataLoader containing training data batches
        val_loader (DataLoader): PyTorch DataLoader containing validation data batches
        input_size (int): Number of input features for the LSTM model
        device (torch.device): Device to run the training on 
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
        patience (int): Number of epochs training will be stopped after with no improvement
    
    Returns:
        model (LSTMWinPredictor): The trained PyTorch model
        train_losses (list): Training loss values for each epoch
        val_losses (list): Validation loss values for each epoch
        train_accuracies (list): Training accuracy values for each epoch
        val_accuracies (list): Validation accuracy values for each epoch
    """

    # Initialize model
    model = NBA_LSTM(input_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
        
        # Stop early if training loss minimizes
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set

    Args:
        model (NBA_LSTM): Trained model
        test_loader (DataLoader): PyTorch DataLoader containing test data
        device (torch.device): Device to run evaluation on (CPU or GPU)
        
    Returns:
        test_accuracy (float): Accuracy metric 
        precision (float): Precision metric
        recall (float): Recall metric
        f1 (float): F1 score 
        all_probabilities (list): Raw prediction probabilities for all test examples
        all_targets (list): Actual target values for all test examples
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
            
            # Store probabilities and targets
            all_probabilities.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate accuracy, precision, recall, f1, confusion
    test_accuracy = test_correct / test_total
    predictions = [1 if p > 0.5 else 0 for p in all_probabilities]
    precision, recall, f1, confusion = calculate_metrics(all_targets, predictions)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion}")
    
    return test_accuracy, precision, recall, f1, all_probabilities, all_targets


def save_model(model, scaler, input_size, test_accuracy, precision, recall, f1):
    """
    Save the trained model, scaler and metadata.
    
    Args:
        model (LSTMWinPredictor): Trained model to save
        scaler: Feature scaler used during preprocessing
        input_size (int, optional): Input size of the model
        test_accuracy (float): Model accuracy 
        precision (float): Precision metric
        recall (float): Recall metric
        f1 (flaot): F1 score
    """
    
    # Where to save model
    model_dir = 'model/saved'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model save path
    team_code = Constants['team'][0]
    seasons = Constants['seasons']
    model_path = os.path.join(model_dir, f"{team_code}_model_{seasons[0]}_to_{seasons[1]}.pth")
    
    # Prepare metadata
    metadata = {
        'team_code': team_code,
        'saved_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_size': input_size,
        'test_accuracy': test_accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'model_config': {
            'hidden_size1': Constants['hidden_size1'],
            'hidden_size2': Constants['hidden_size2'],
            'num_layers': Constants['num_layers'],
            'dropout': Constants['dropout']
        }
    }
    
    # Save model, scaler, and metadata
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'metadata': metadata
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Also save latest model for easier reference
    latest_path = os.path.join(model_dir, f"{team_code}_model_latest.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'metadata': metadata
    }, latest_path)
    
    print(f"Latest model saved to {latest_path}")
