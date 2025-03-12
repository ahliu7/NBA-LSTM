import torch
import numpy as np
import pandas as pd
import os
from model.model import NBA_LSTM
from config import Constants


def load_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        tuple: (model, scaler, metadata)
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract components
    model_state_dict = checkpoint['model_state_dict']
    scaler = checkpoint['scaler']
    metadata = checkpoint.get('metadata', {})
    
    # Get input size
    input_size = metadata.get('input_size')
    if input_size is None:
        raise ValueError("Model metadata doesn't contain input_size")
    
    # Create model
    model = NBA_LSTM(input_size=input_size)
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()  # Set model to evaluation mode
    
    print(f"Model loaded from {model_path}")
    
    return model, scaler, metadata


def predict_next_game(team_code, sequence_length=5, model_path=None):
    """
    Predict if the team will win their next game based on their recent performance.
    
    Args:
        team_code (str): Team code (e.g., 'lal')
        sequence_length (int): Number of recent games to use for prediction
        model_path (str, optional): Path to the model file
        
    Returns:
        dict: Prediction result with win probability
    """
    # Determine model path if not provided
    if model_path is None:
        model_path = f"model/saved/{team_code}_model_latest.pth"
    
    # Load model
    model, scaler, _ = load_model(model_path)
    
    # Load processed data
    data_path = f"data/datasets/{team_code}_processed_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Get the most recent games
    feature_cols = [col for col in df.columns 
                   if col not in ['Date', 'Season', 'Team', 'Opp', 'Win']]
    
    recent_games = df.tail(sequence_length)[feature_cols].values
    
    # Scale the features
    scaled_data = scaler.transform(recent_games)
    
    # Convert to tensor
    inputs = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        win_prob = model(inputs).item()
    
    # Get the most recent game info for context
    last_game = df.iloc[-1]
    opponent = last_game.get('Opp', 'Unknown')
    
    # Prepare result
    result = {
        'team': team_code.upper(),
        'win_probability': win_prob,
        'prediction': 'Win' if win_prob > 0.5 else 'Loss'
    }
    
    return result


if __name__ == "__main__":
    # Get team code from user input
    team_code = Constants['team'][0]

    # Predict
    result = predict_next_game(team_code)
    
    # Display results
    print(f"\nPrediction for {result['team']} next game:")
    print(f"Win Probability: {result['win_probability']:.2%}")
    print(f"Prediction: {result['prediction']}")
