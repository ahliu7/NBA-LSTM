import pandas as pd
import time
import random
from config import Constants

def scrape_team_data(team, seasons):
    """
    Scrape game data for the specified team over specified seasons.
    
    Args:
        team (str): Three-letter team code (e.g., 'LAL' for Lakers)
        seasons (tuple): start and end seasons to scrape (e.g., (2022, 2023))
        
    Returns:
        DataFrame: Combined game data for the specified team
    """

    print(f"Scraping data for {team.upper()} from {seasons[0]}-{seasons[-1]}...")

    stats = Constants['stats']
    
    team_stats = {stat: f"Team_{stat}" for stat in stats}
    opp_stats = {stat + '.1': f"Opp_{stat}" for stat in stats}

    team_df = pd.DataFrame()

    for season in seasons:
        url = f"https://www.basketball-reference.com/teams/{team}/{season}/gamelog/"
        print(f"Fetching: {url}")

        try:
            tables = pd.read_html(url, header=1)
            season_df = None

            for table in tables:
                if 'Rk' in table.columns:  # Ensure the correct table is selected
                    season_df = table
                    break

            if season_df is None:
                raise ValueError("No valid game log table found.")

            season_df = season_df[season_df['Rk'].astype(str).str.isnumeric()]
            season_df = season_df.drop(columns=['Rk'], errors='ignore')

            season_df = season_df.rename(columns={'Unnamed: 3': 'Home', 'Tm': 'Team_pts', 'Opp.1': 'Opp_pts'})
            season_df = season_df.rename(columns=team_stats)
            season_df = season_df.rename(columns=opp_stats)

            season_df['Home'] = season_df['Home'].apply(lambda x: 'A' if x == '@' else 'H')

            season_df.insert(0, 'Season', season)
            season_df.insert(1, 'Team', team.upper())

            team_df = pd.concat([team_df, season_df], ignore_index=True)

            time.sleep(random.randint(4, 6))

        except Exception as e:
            print(f"Error scraping {season} data for {team}: {e}")

    print(f"Scraped {len(team_df)} games for {team.upper()}")
    return team_df


