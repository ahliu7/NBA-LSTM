import pandas as pd
import random
import time

teams = [
    'atl', 'bos', 'brk', 'cho', 'chi', 'cle', 'dal', 'den', 'det', 'gsw', 'hou', 'ind', 'lac', 'lal', 'mem', 
    'mia', 'mil', 'min', 'nop', 'nyk', 'okc', 'orl', 'phi', 'pho', 'por', 'sac', 'sas', 'tor', 'uta', 'was']
len(teams)

seasons = list(range(2019,2024))
seasons

stats = [
    'FG', 'FGA', 'FG%', 
    '3P', '3PA', '3P%',
    'FT', 'FTA', 'FT%',
    'ORB', 'TRB', 'AST', 
    'STL', 'BLK', 'TOV', 'PF' ]

team_stats = {stat: f"Team_{stat}" for stat in stats}
opp_stats = {stat + '.1': f"Opp_{stat}" for stat in stats}

nba_df = pd.DataFrame()

for season in seasons:
    for team in teams:
        url = f"https://www.basketball-reference.com/teams/{team}/{season}/gamelog/"
        print(url)

        team_df = pd.read_html(url, header=1, attrs={'id': 'tgl_basic'})[0]
        team_df = team_df[(team_df['Rk'].str != '') & (team_df['Rk'].str.isnumeric())]
        team_df = team_df.drop(columns=['Rk', 'Unnamed: 24'])

        team_df = team_df.rename(columns={'Unnamed: 3':'Home', 'Tm':'Team_pts', 'Opp.1':'Opp_pts'})
        team_df = team_df.rename(columns=team_stats)
        team_df = team_df.rename(columns=opp_stats)

        team_df['Home'] = team_df['Home'].apply(lambda x: 'A' if x=='@' else 'H')

        team_df.insert(0, 'Season', season)
        team_df.insert(1, 'Team', team.upper())

        nba_df = pd.concat([nba_df, team_df], ignore_index=True)

        time.sleep(random.randint(4,6))

print(nba_df)
