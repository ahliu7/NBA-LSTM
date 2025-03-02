import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocess the raw game data for model training
    
    Args:
        df (DataFrame): Raw game data
    
    Returns:
        DataFrame: Preprocessed data ready for feature engineering
    """

    processed_df = df.copy()

    numeric_columns = [
        'Team_pts', 'Opp_pts', 'Team_FG', 'Team_3P', 'Team_FGA', 'Opp_FG', 'Opp_3P', 
        'Opp_FGA', 'Team_TOV', 'Team_FTA', 'Team_ORB', 'Opp_ORB', 'Opp_TRB', 'Team_AST', 'Opp_AST'
    ]
    for col in numeric_columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    processed_df = processed_df.sort_values('Date')
    processed_df['Rslt'] = processed_df['Rslt'].apply(lambda x: 1 if x == 'W' else 0)
    processed_df = processed_df.rename(columns={'Rslt': 'Win'})
    processed_df['Point_Diff'] = processed_df['Team_pts'] - processed_df['Opp_pts']
    processed_df['eFG%'] = (processed_df['Team_FG'] + 0.5 * processed_df['Team_3P']) / processed_df['Team_FGA']
    processed_df['Opp_eFG%'] = (processed_df['Opp_FG'] + 0.5 * processed_df['Opp_3P']) / processed_df['Opp_FGA']
    processed_df['TOV_Rate'] = processed_df['Team_TOV'] / (processed_df['Team_FGA'] + 0.44 * processed_df['Team_FTA'] + processed_df['Team_TOV'])
    processed_df['ORB%'] = processed_df['Team_ORB'] / (processed_df['Team_ORB'] + (processed_df['Opp_TRB'] - processed_df['Opp_ORB']))
    processed_df['AST_TO_Ratio'] = processed_df['Team_AST'] / processed_df['Team_TOV']
    processed_df['Home'] = (processed_df['Home'] == 'H').astype(int)
    
    # Calculate rolling averages for last 5 games
    rolling_columns = ['Team_pts', 'Opp_pts', 'Point_Diff', 'eFG%', 'Opp_eFG%', 
                       'TOV_Rate', 'ORB%', 'AST_TO_Ratio', 'Win']
    for col in rolling_columns:
        processed_df[f'{col}_5game_avg'] = processed_df[col].rolling(window=5, min_periods=1).mean()
    
    # Calculate win percentage in last 10 games
    processed_df['Win_10game_pct'] = processed_df['Win'].rolling(window=10, min_periods=1).mean()

    # Fill in NaN values with mean
    processed_df = processed_df.drop(columns=['OT'])
    processed_df = processed_df.fillna(processed_df.select_dtypes(include=['number']).mean())

    return processed_df


def create_sequences(data, sequence_length=5, target_col='Win'):
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


class GameSequenceDataset(Dataset):
    """PyTorch Dataset for game sequences"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_model_data(processed_df, sequence_length=5, test_size=0.2, val_size=0.15):
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

    # Drop non-numeric columns and date
    data = processed_df.drop(columns=['Date', 'Season', 'Team', 'Opp'])
    
    # Scale the features
    feature_data = data.drop(columns=['Win'])
    target_data = data['Win']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Put scaled features back into a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=feature_data.columns)
    scaled_df['Win'] = target_data.values
    
    # Create sequences
    X, y = create_sequences(scaled_df, sequence_length)
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_proportion, random_state=42, shuffle=False
    )
    
    # Create datasets
    train_dataset = GameSequenceDataset(X_train, y_train)
    val_dataset = GameSequenceDataset(X_val, y_val)
    test_dataset = GameSequenceDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get input size (number of features)
    input_size = X_train.shape[2]
    
    return train_loader, val_loader, test_loader, scaler, input_size


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
    
    plt.tight_layout()
    plt.savefig(f"{team_name.lower()}_performance.png")
    plt.show()
