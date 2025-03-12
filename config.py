# Global parameters
Constants = { 
    
    # Target team and seasons
    'team': ('den', 'Nuggets'), 
    'seasons': (2022, 2025),

    # Teams and stats
    'teams': [
        'atl', 'bos', 'brk', 'cho', 'chi', 'cle', 'dal', 'den', 'det', 'gsw', 'hou', 'ind', 'lac', 'lal', 'mem', 
        'mia', 'mil', 'min', 'nop', 'nyk', 'okc', 'orl', 'phi', 'pho', 'por', 'sac', 'sas', 'tor', 'uta', 'was'
    ],

    'stats': [
        'FG', 'FGA', 'FG%', 
        '3P', '3PA', '3P%',
        'FT', 'FTA', 'FT%',
        'ORB', 'TRB', 'AST', 
        'STL', 'BLK', 'TOV', 'PF'
    ],

    # LSTM Model parameters
    'hidden_size1': 64, 
    'hidden_size2': 32, 
    'num_layers': 2, 
    'dropout': 0.5,

    # For preparing model data
    'sequence_length': 3,
    'test_size': 0.2,
    'val_size': 0.5,

    # For creating data loaders
    'batch_size': 32,

    # For training LSTM
    'epochs': 500,
    'learning_rate': 0.0001,
    'patience': 50
}