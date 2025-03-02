from Constants import Constants
from NewScraper import scrape_team_data
from Preprocess import preprocess_data, prepare_model_data, plot_team_performance

if __name__ == "__main__":

    # Specifiy team, seasons to train model
    team_code = 'lal'
    team_name = 'Lakers' 
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
    train_loader, val_loader, test_loader, scaler, input_size = prepare_model_data(processed_data)
    
    print(f"Data prepared for modeling:")
    print(f"- Input size: {input_size}")
    print(f"- Training batches: {len(train_loader)}")
    print(f"- Validation batches: {len(val_loader)}")
    print(f"- Test batches: {len(test_loader)}")
    
    # Next step would be to train the LSTM model using the data loaders
    print("\nData is ready for model training. Import the LSTM model from the previous script to complete the workflow.")
