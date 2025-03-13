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
