import torch
import torch.nn as nn
from config import Constants
from data.new_scraper import scrape_team_data
from data.cleaning import preprocess_data
from data.transformation import prepare_model_data
from visualization.visualize import plot_team_performance, plot_training_history, plot_roc_curve
from training import train_model, evaluate_model, save_model

if __name__ == "__main__":

    # Specifiy team, seasons to train model
    team_code = Constants['team'][0]
    team_name = Constants['team'][1]
    seasons_to_scrape = list(range(Constants['seasons'][0], Constants['seasons'][1]+1))
    
    # Scrape data
    team_data = scrape_team_data(team_code, seasons_to_scrape)
    team_data.to_csv(f"{team_code}_raw_data.csv", index=False)
    print(f"Raw data saved to {team_code}_raw_data.csv")

    # Preprocess raw data
    processed_data = preprocess_data(team_data)
    processed_data.to_csv(f"{team_code}_processed_data.csv", index=False)
    print(f"Processed data saved to {team_code}_processed_data.csv")
    
    # Visualize team performance
    plot_team_performance(processed_data, team_name)
    
    # Prepare data for modeling
    sequence_length = Constants['squence_length']
    test_size = Constants['tests_size']
    val_size = Constants['val_size']
    train_loader, val_loader, test_loader, scaler, input_size = prepare_model_data(processed_data, sequence_length, test_size, val_size)
    
    print(f"Data ready for modeling:")
    print(f"Input size: {input_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Start training

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    print("Training starts...")
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        train_loader, val_loader, input_size, device, 
        epochs=Constants['epochs'], learning_rate=Constants['learning_rate'], patience=Constants['patience']
    )
    print("Training complete")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate model
    print('Evaluation starts...')
    test_accuracy, precision, recall, f1, probabilities, targets = evaluate_model(model, test_loader, device)
    print('Evaluation complete')

    # Plot model evaluation 
    auc_score = plot_roc_curve(y_true=targets, y_probas=probabilities)
    print(f"AUC Score: {auc_score:.4f}")
   
    # Save model
    save_model(model, scaler, input_size, test_accuracy, precision, recall, f1)
