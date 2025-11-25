import pandas as pd
import os
import numpy as np
import joblib
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

folder_path = '/Users/thevedantsingh/Desktop/Tennis'
dataframes = []

# Load data
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') and not file_name.startswith('~'):
        try:
            year = int(file_name.split('.')[0])
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path, engine='openpyxl')
            df['Year_File'] = year # Renamed to avoid conflict with Date column
            dataframes.append(df)
        except ValueError:
            continue

final_df = pd.concat(dataframes, ignore_index=True)

# Preprocessing
final_df['Date'] = pd.to_datetime(final_df['Date'])
final_df = final_df.sort_values('Date').reset_index(drop=True)

# --- Feature Engineering: Elo, Fatigue, H2H, Form ---
elo_rating = defaultdict(lambda: 1500)
surface_elo = defaultdict(lambda: defaultdict(lambda: 1500))
last_match_date = {}

# H2H and Form History
h2h_wins = defaultdict(lambda: defaultdict(int)) # h2h_wins[p1][p2] = wins
match_history = defaultdict(list) # match_history[p1] = [(date, win/loss, surface), ...]

K_FACTOR = 32

winner_elos = []
loser_elos = []
winner_surface_elos = []
loser_surface_elos = []
winner_days_since = []
loser_days_since = []

# New Feature Lists
w_h2h = []
l_h2h = []
w_last10 = []
l_last10 = []
w_last10_surf = []
l_last10_surf = []

print("Calculating Elo, Fatigue, H2H, and Form...")

for idx, row in final_df.iterrows():
    w_id = row['Winner']
    l_id = row['Loser']
    surface = row['Surface']
    date = row['Date']
    
    # --- Elo & Fatigue (Existing) ---
    w_elo = elo_rating[w_id]
    l_elo = elo_rating[l_id]
    w_surf_elo = surface_elo[surface][w_id]
    l_surf_elo = surface_elo[surface][l_id]
    
    winner_elos.append(w_elo)
    loser_elos.append(l_elo)
    winner_surface_elos.append(w_surf_elo)
    loser_surface_elos.append(l_surf_elo)
    
    if w_id in last_match_date:
        winner_days_since.append((date - last_match_date[w_id]).days)
    else:
        winner_days_since.append(30)
        
    if l_id in last_match_date:
        loser_days_since.append((date - last_match_date[l_id]).days)
    else:
        loser_days_since.append(30)
        
    last_match_date[w_id] = date
    last_match_date[l_id] = date
    
    # --- H2H (New) ---
    w_h2h.append(h2h_wins[w_id][l_id])
    l_h2h.append(h2h_wins[l_id][w_id])
    
    # --- Recent Form (Last 10 Matches) ---
    def get_form(player, current_surface, n=10):
        history = match_history[player]
        if not history:
            return 0.5, 0.5 # Default if no history
            
        recent = history[-n:]
        wins = sum(1 for x in recent if x[1] == 1)
        win_pct = wins / len(recent)
        
        # Surface Form
        surf_history = [x for x in history if x[2] == current_surface]
        if not surf_history:
            surf_win_pct = 0.5
        else:
            recent_surf = surf_history[-n:]
            surf_wins = sum(1 for x in recent_surf if x[1] == 1)
            surf_win_pct = surf_wins / len(recent_surf)
            
        return win_pct, surf_win_pct

    w_form, w_surf_form = get_form(w_id, surface)
    l_form, l_surf_form = get_form(l_id, surface)
    
    w_last10.append(w_form)
    l_last10.append(l_form)
    w_last10_surf.append(w_surf_form)
    l_last10_surf.append(l_surf_form)
    
    # --- Updates (After extracting features) ---
    
    # Update Elo
    E_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
    E_w_surf = 1 / (1 + 10 ** ((l_surf_elo - w_surf_elo) / 400))
    
    elo_rating[w_id] = w_elo + K_FACTOR * (1 - E_w)
    elo_rating[l_id] = l_elo + K_FACTOR * (0 - (1 - E_w))
    
    surface_elo[surface][w_id] = w_surf_elo + K_FACTOR * (1 - E_w_surf)
    surface_elo[surface][l_id] = l_surf_elo + K_FACTOR * (0 - (1 - E_w_surf))
    
    # Update H2H
    h2h_wins[w_id][l_id] += 1
    
    # Update Match History (1 for Win, 0 for Loss)
    match_history[w_id].append((date, 1, surface))
    match_history[l_id].append((date, 0, surface))

# Assign new columns
final_df['Winner_Elo'] = winner_elos
final_df['Loser_Elo'] = loser_elos
final_df['Winner_Surface_Elo'] = winner_surface_elos
final_df['Loser_Surface_Elo'] = loser_surface_elos
final_df['Winner_Days_Since'] = winner_days_since
final_df['Loser_Days_Since'] = loser_days_since

final_df['Winner_H2H'] = w_h2h
final_df['Loser_H2H'] = l_h2h
final_df['Winner_Last10'] = w_last10
final_df['Loser_Last10'] = l_last10
final_df['Winner_Last10_Surf'] = w_last10_surf
final_df['Loser_Last10_Surf'] = l_last10_surf

# --- Standard Preprocessing (Copied & Adapted from data.py) ---

# One-hot encode Surface/Court
surface_dummies = pd.get_dummies(final_df['Surface'], prefix='Surface')
court_dummies = pd.get_dummies(final_df['Court'], prefix='Court')
final_df = pd.concat([final_df, surface_dummies, court_dummies], axis=1)
final_df = final_df.drop(['Surface', 'Court'], axis=1)

# Encode Series
series_mapping = {'ATP250': 250, 'Grand Slam': 2000, 'ATP500': 500, 'Masters 1000': 1000, 'Masters Cup': 1000}
final_df['Series_encoded'] = final_df['Series'].map(series_mapping).fillna(250) / 250

# Encode Round
round_mapping = {'1st Round': 0, '2nd Round': 1, '3rd Round': 2, '4th Round': 3, 'Quarterfinals': 4, 'Semifinals': 5, 'The Final': 6, 'Final': 6, 'Round Robin': 0}
final_df['Round_encoded'] = final_df['Round'].map(round_mapping).fillna(0).astype(int)

# Clean Odds
odds_columns = ['B365W', 'B365L', 'PSW', 'PSL']
final_df = final_df.dropna(subset=odds_columns, how='all')

# Fill missing odds
for col in ['PSW', 'PSL', 'EXW', 'EXL', 'LBW', 'LBL', 'SJW','SJL']:
    if col in final_df.columns:
        if col.endswith('W'): final_df[col] = final_df[col].fillna(final_df['AvgW'])
        elif col.endswith('L'): final_df[col] = final_df[col].fillna(final_df['AvgL'])

final_df = final_df.dropna(subset=['AvgW', 'WRank', 'LRank'])
final_df['Best of'] = final_df['Best of'].fillna(3)
final_df = final_df[final_df['Comment'] == 'Completed']

# Drop unneeded columns
cols_to_drop = ['ATP', 'Location', 'Tournament', 'Series', 'Round', 'Winner', 'Loser', 'Comment', 'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'Wsets', 'Lsets']
final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns])

# Convert bools
for col in final_df.select_dtypes(include=['bool']).columns:
    final_df[col] = final_df[col].astype(int)

# --- Row Swapping (Target Creation) ---
final_df['Win'] = 1
# Create deterministic ID for swapping (using Rank or Name hash if needed, but here we use index to be random-ish but consistent)
np.random.seed(42)
mask = np.random.rand(len(final_df)) > 0.5

# Swap logic
swap_cols = [
    ('WPts', 'LPts'), ('WRank', 'LRank'), 
    ('PSW', 'PSL'), ('MaxW', 'MaxL'), ('AvgW', 'AvgL'), 
    ('Winner_Elo', 'Loser_Elo'), ('Winner_Surface_Elo', 'Loser_Surface_Elo'),
    ('Winner_Days_Since', 'Loser_Days_Since'),
    ('Winner_H2H', 'Loser_H2H'),
    ('Winner_Last10', 'Loser_Last10'),
    ('Winner_Last10_Surf', 'Loser_Last10_Surf')
]

for w_col, l_col in swap_cols:
    if w_col in final_df.columns and l_col in final_df.columns:
        final_df.loc[mask, [w_col, l_col]] = final_df.loc[mask, [l_col, w_col]].values

# Set Target
final_df.loc[mask, 'Win'] = 0

# Rename columns to P1/P2 generic
rename_map = {
    'WPts': 'P1_Pts', 'LPts': 'P2_Pts',
    'WRank': 'P1_Rank', 'LRank': 'P2_Rank',
    'Winner_Elo': 'P1_Elo', 'Loser_Elo': 'P2_Elo',
    'Winner_Surface_Elo': 'P1_Surface_Elo', 'Loser_Surface_Elo': 'P2_Surface_Elo',
    'Winner_Days_Since': 'P1_Days_Since', 'Loser_Days_Since': 'P2_Days_Since',
    'Winner_H2H': 'P1_H2H', 'Loser_H2H': 'P2_H2H',
    'Winner_Last10': 'P1_Last10', 'Loser_Last10': 'P2_Last10',
    'Winner_Last10_Surf': 'P1_Last10_Surf', 'Loser_Last10_Surf': 'P2_Last10_Surf'
}
final_df = final_df.rename(columns=rename_map)

# Feature Engineering 2
final_df['Elo_Diff'] = final_df['P1_Elo'] - final_df['P2_Elo']
final_df['Surface_Elo_Diff'] = final_df['P1_Surface_Elo'] - final_df['P2_Surface_Elo']
final_df['Rank_Diff'] = final_df['P2_Rank'] - final_df['P1_Rank'] # Higher rank is lower number
final_df['Pts_Diff'] = final_df['P1_Pts'] - final_df['P2_Pts']
final_df['H2H_Diff'] = final_df['P1_H2H'] - final_df['P2_H2H']
final_df['Form_Diff'] = final_df['P1_Last10'] - final_df['P2_Last10']
final_df['Surf_Form_Diff'] = final_df['P1_Last10_Surf'] - final_df['P2_Last10_Surf']

# Fill NaNs
final_df = final_df.fillna(0)

# Scale
scaler = MinMaxScaler()
scale_cols = [
    'P1_Pts', 'P2_Pts', 'P1_Elo', 'P2_Elo', 'P1_Surface_Elo', 'P2_Surface_Elo', 
    'Elo_Diff', 'Surface_Elo_Diff', 'H2H_Diff', 'Form_Diff', 'Surf_Form_Diff',
    'P1_H2H', 'P2_H2H', 'P1_Last10', 'P2_Last10'
]
final_df[scale_cols] = scaler.fit_transform(final_df[scale_cols])
joblib.dump(scaler, 'enhanced_scaler.pkl')

# Save Elo and Player Data for Web App
print("Saving Elo and Player Data...")
# Convert defaultdicts to dicts for pickling
# joblib.dump(dict(player_id_mapping), 'player_id_mapping.pkl') # Not needed, using names
joblib.dump(dict(elo_rating), 'elo_rating.pkl')
# surface_elo is a nested defaultdict, convert to dict of dicts
surface_elo_dict = {k: dict(v) for k, v in surface_elo.items()}
joblib.dump(surface_elo_dict, 'surface_elo.pkl')
joblib.dump(last_match_date, 'last_match_date.pkl')
# Also save H2H and Match History for web app
h2h_wins_dict = {k: dict(v) for k, v in h2h_wins.items()}
joblib.dump(h2h_wins_dict, 'h2h_wins.pkl')
joblib.dump(dict(match_history), 'match_history.pkl')
print("Data saved successfully.")

# Extract Year for splitting
final_df['Year'] = final_df['Date'].dt.year

# Split
train = final_df[final_df['Year'] <= 2024]
test = final_df[final_df['Year'] == 2025]
val = test

print(f"Enhanced Data Prepared. Train: {train.shape}, Test: {test.shape}")
