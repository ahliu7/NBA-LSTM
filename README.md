# NBA LSTM Win Predictor

A project in deep learning: Predicting NBA game outcomes using LSTM neural networks to model time series data.

## Overview

This project uses Long Short-Term Memory (LSTM) neural networks to predict whether an NBA team will win their next game based on historical game statistics. By analyzing sequences of previous games, the model learns patterns in team performance that can indicate future success.

## Features

- **Data ETL**: Scrapes and preprocesses NBA game statistics into usable features
- **LSTM Model**: Implements PyTorch LSTM architecture for time series prediction
- **Training Pipeline**: Complete pipeline for model training with early stopping and evaluation
- **Win Prediction**: Predicts the probability of a team winning their next game
- **Visualization**: Plots training metrics and team performance trends

## Repository Structure

```
NBA-LSTM/
├── config.py                 # Configuration parameters for the project
├── main.py                   # Main script for training the model
├── predict.py                # Script for making predictions with trained models
├── training.py               # Model training and evaluation functions
├── utils.py                  # Utility functions
├── data/
│   ├── cleaning.py           # Data cleaning and preprocessing
│   ├── new_scraper.py        # Web scraper for NBA game statistics
│   ├── transformation.py     # Feature transformation and sequence creation
│   └── datasets/             # Directory for storing scraped and processed data
├── model/
│   ├── model.py              # LSTM model architecture definition
│   └── saved/                # Directory for storing trained models
└── visualization/            # Scripts for data and model performance visualization
```

## How It Works

### Data Collection

The system scrapes game statistics from [Basketball Reference](https://www.basketball-reference.com/) for a specific team across selected seasons. This includes points, rebounds, assists, shooting percentages, and more for both the team and opponents.

### Preprocessing

Raw data undergoes transformation to create predictive features:

- Basic stats normalization (points, rebounds, etc.)
- Calculation of advanced metrics (effective field goal percentage, turnover rate)
- Creation of rolling statistics to capture recent performance
- Home/away game encoding

### Sequence Creation

Individual games are organized into sequences of specified length, allowing the model to learn from patterns in performance over time rather than isolated games.

### LSTM Architecture

The PyTorch model uses a two-layer LSTM network with dropout regularization, processing these game sequences to identify temporal patterns associated with wins and losses.

### Training

The model trains using binary cross-entropy loss, with scheduled evaluation on a validation set. Early stopping prevents overfitting by preserving the model state that performs best on validation data.

### Prediction

For predicting upcoming games, the model analyzes the team's most recent game sequence and outputs a win probability. This process requires only historical data, making it suitable for real-time prediction.

## Usage Instructions

### Configuration

All model and data parameters are stored in `config.py`. The main parameters you'll want to modify are:

```python
Constants = {
    # Target team and seasons
    'team': ('den', 'Nuggets'),  # Team code and name
    'seasons': (2022, 2025),     # Range of seasons to scrape (inclusive)

    # LSTM parameters
    'sequence_length': 3,        # Number of games in each sequence
    'hidden_size1': 64,          # Size of first LSTM layer
    'hidden_size2': 32,          # Size of second LSTM layer
    'dropout': 0.5,              # Dropout rate for regularization

    # Training parameters
    'epochs': 500,               # Maximum number of training epochs
    'learning_rate': 0.0001,     # Learning rate for optimizer
    'patience': 50               # Early stopping patience
}
```

### Training a Model

To train a model for a specific team:

1. Edit `config.py` to set your desired team and seasons:

   ```python
   'team': ('lal', 'Lakers'),  # Change to your team of interest
   'seasons': (2020, 2023),    # Change to desired seasons
   ```

2. Run the training script:
   ```bash
   python main.py
   ```

This will:

- Scrape data for the specified team and seasons
- Preprocess the data and create sequences
- Train the LSTM model with early stopping
- Evaluate the model on test data
- Save the trained model to `model/saved/{team_code}_model_latest.pth`

### Making Predictions

To predict if a team will win their next game:

1. Edit `config.py` to set your desired team and seasons:

   ```python
   'team': ('lal', 'Lakers'),  # Change to your team of interest
   'seasons': (2020, 2023),    # Change to desired seasons
   ```

2. Ensure you have a trained model for that specific team
3. Run the prediction script:
   ```bash
   python predict.py
   ```

## Model Performance

The model achieves:

- **Test Accuracy**: ~64%
- **F1 Score**: ~0.78
- **AUC**: ~0.73

Performance varies by team and can be improved with more data and feature engineering.

## Future Improvements

- Add player-specific features (injuries, rest days)
- Incorporate head-to-head matchup statistics
- Implement ensemble methods to improve prediction accuracy
- Add support for playoff-specific predictions
