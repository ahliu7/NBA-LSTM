# Global parameters
Constants = { 
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

    # Target team and seasons
    'team': ('lal', 'Lakers'), 
    'seasons': (2023, 2024),

    # For preparing model data
    'sequence_length': 5,
    'test_size': 0.2,
    'val_size': 0.5,

    # For creating data loaders
    'batch_size': 32,

    # For training LSTM
    'epochs': 100,
    'learning_rate': 0.001,
    'patience': 10
}