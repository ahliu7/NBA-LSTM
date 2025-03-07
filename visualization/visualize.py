import matplotlib.pyplot as plt

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
    
    # Plot
    plt.tight_layout()
    plt.savefig(f"{team_name.lower()}_performance.png")
    plt.show()
