import pandas as pd
import numpy as np
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

    # Make copy of dataframe
    processed_df = df.copy()

    # Convert each column to numeric
    numeric_columns = [
        'Team_pts', 'Opp_pts', 'Team_FG', 'Team_3P', 'Team_FGA', 'Opp_FG', 'Opp_3P', 
        'Opp_FGA', 'Team_TOV', 'Team_FTA', 'Team_ORB', 'Opp_ORB', 'Opp_TRB', 'Team_AST', 'Opp_AST'
    ]
    for col in numeric_columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    # Process specific columns and add extra statistics
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
