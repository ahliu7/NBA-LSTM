import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Constants

class GameSequenceDataset(Dataset):
    """Custome PyTorch Dataset for game sequences to better model time series data"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, sequence_length, target_col='Win'):
    """
    Convert DataFrame to sequences for LSTM model
    
    Args:
        data (DataFrame): Preprocessed data
        sequence_length (int): Number of games in each sequence
        target_col (str): Column to use as prediction target
    
    Returns:
        tuple: (X, y) arrays for model training
    """

    # Get feature columns (all except target)
    feature_cols = [col for col in data.columns if col != target_col]
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Get sequence of games
        seq = data.iloc[i:i+sequence_length][feature_cols].values
        # Get target of the next game
        target = data.iloc[i+sequence_length][target_col]
        
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y)


def prepare_model_data(processed_df, sequence_length, test_size, val_size):
    """
    Prepare data for LSTM model training
    
    Args:
        processed_df (DataFrame): Preprocessed game data
        sequence_length (int): Number of games in each sequence
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of remaining data for validation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler, input_size)
    """

    # Drop non-numeric columns
    data = processed_df.drop(columns=['Date', 'Season', 'Team', 'Opp'])
    
    # Scale features
    feature_data = data.drop(columns=['Win'])
    target = data['Win']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Put scaled features back into a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=feature_data.columns)
    scaled_df['Win'] = target.values
    
    # Create sequences
    X, y = create_sequences(scaled_df, sequence_length)
    
    # Split data and create custome game sequence data sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_proportion, random_state=42, shuffle=False
    )
    
    train_dataset = GameSequenceDataset(X_train, y_train)
    val_dataset = GameSequenceDataset(X_val, y_val)
    test_dataset = GameSequenceDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = Constants['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get input size 
    input_size = X_train.shape[2]
    
    return train_loader, val_loader, test_loader, scaler, input_size

    