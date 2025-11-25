import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load Artifacts
print("Loading model and data...")
model = joblib.load('xgb_enhanced.joblib')
scaler = joblib.load('enhanced_scaler.pkl')
elo_rating = joblib.load('elo_rating.pkl')
surface_elo = joblib.load('surface_elo.pkl')
last_match_date = joblib.load('last_match_date.pkl')
h2h_wins = joblib.load('h2h_wins.pkl')
match_history = joblib.load('match_history.pkl')

# Constants
K_FACTOR = 32
THRESHOLD = 0.65
MIN_ODDS = 1.7

def get_input(prompt, type_=str):
    while True:
        try:
            val = input(prompt)
            if type_ == str:
                return val.strip()
            return type_(val)
        except ValueError:
            print("Invalid input. Please try again.")

def get_form(player, current_surface, n=10):
    history = match_history.get(player, [])
    if not history:
        return 0.5, 0.5 # Default
        
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

def predict_match():
    print("\n" + "="*40)
    print("🎾  TENNIS MATCH PREDICTOR  🎾")
    print("="*40)
    
    # Inputs
    p1_name = get_input("Player 1 Name (e.g. Novak Djokovic): ")
    p2_name = get_input("Player 2 Name (e.g. Jannik Sinner): ")
    
    # Check if players exist
    if p1_name not in elo_rating:
        print(f"WARNING: {p1_name} not found in database. Using default stats.")
    if p2_name not in elo_rating:
        print(f"WARNING: {p2_name} not found in database. Using default stats.")
        
    surface = get_input("Surface (Hard, Clay, Grass): ")
    if surface not in ['Hard', 'Clay', 'Grass']:
        print("Invalid surface. Defaulting to Hard.")
        surface = 'Hard'
        
    court = get_input("Court (Outdoor, Indoor): ")
    if court not in ['Outdoor', 'Indoor']:
        print("Invalid court. Defaulting to Outdoor.")
        court = 'Outdoor'
        
    best_of = get_input("Best of (3 or 5): ", int)
    
    p1_rank = get_input("Player 1 Rank: ", int)
    p2_rank = get_input("Player 2 Rank: ", int)
    
    p1_pts = get_input("Player 1 Points: ", int)
    p2_pts = get_input("Player 2 Points: ", int)
    
    p1_odds = get_input("Player 1 Odds: ", float)
    p2_odds = get_input("Player 2 Odds: ", float)
    
    round_num = get_input("Round (1-6, 6=Final): ", int)
    series_val = get_input("Series (250, 500, 1000, 2000): ", int)
    
    # --- Feature Engineering ---
    
    # Elo
    p1_elo = elo_rating.get(p1_name, 1500)
    p2_elo = elo_rating.get(p2_name, 1500)
    
    p1_surf_elo = surface_elo.get(surface, {}).get(p1_name, 1500)
    p2_surf_elo = surface_elo.get(surface, {}).get(p2_name, 1500)
    
    # Fatigue
    today = pd.Timestamp.now()
    p1_last = last_match_date.get(p1_name, today - pd.Timedelta(days=30))
    p2_last = last_match_date.get(p2_name, today - pd.Timedelta(days=30))
    
    p1_days = (today - p1_last).days
    p2_days = (today - p2_last).days
    
    # H2H
    p1_h2h = h2h_wins.get(p1_name, {}).get(p2_name, 0)
    p2_h2h = h2h_wins.get(p2_name, {}).get(p1_name, 0)
    
    # Form
    p1_form, p1_surf_form = get_form(p1_name, surface)
    p2_form, p2_surf_form = get_form(p2_name, surface)
    
    # Diffs
    elo_diff = p1_elo - p2_elo
    surf_elo_diff = p1_surf_elo - p2_surf_elo
    rank_diff = p2_rank - p1_rank
    h2h_diff = p1_h2h - p2_h2h
    form_diff = p1_form - p2_form
    surf_form_diff = p1_surf_form - p2_surf_form
    
    # Create DataFrame
    data = {
        'P1_Elo': [p1_elo], 'P2_Elo': [p2_elo],
        'P1_Surface_Elo': [p1_surf_elo], 'P2_Surface_Elo': [p2_surf_elo],
        'Elo_Diff': [elo_diff], 'Surface_Elo_Diff': [surf_elo_diff],
        'P1_Days_Since': [p1_days], 'P2_Days_Since': [p2_days],
        'Rank_Diff': [rank_diff], 'P1_Rank': [p1_rank], 'P2_Rank': [p2_rank],
        'P1_Pts': [p1_pts], 'P2_Pts': [p2_pts],
        'P1_H2H': [p1_h2h], 'P2_H2H': [p2_h2h], 'H2H_Diff': [h2h_diff],
        'P1_Last10': [p1_form], 'P2_Last10': [p2_form], 'Form_Diff': [form_diff],
        'P1_Last10_Surf': [p1_surf_form], 'P2_Last10_Surf': [p2_surf_form], 'Surf_Form_Diff': [surf_form_diff],
        'Surface_Clay': [1 if surface == 'Clay' else 0],
        'Surface_Grass': [1 if surface == 'Grass' else 0],
        'Surface_Hard': [1 if surface == 'Hard' else 0],
        'Court_Indoor': [1 if court == 'Indoor' else 0],
        'Court_Outdoor': [1 if court == 'Outdoor' else 0],
        'Best of': [best_of],
        'Round_encoded': [round_num],
        'Series_encoded': [series_val / 250.0]
    }
    
    df = pd.DataFrame(data)
    
    # Scale
    scale_cols = [
        'P1_Pts', 'P2_Pts', 'P1_Elo', 'P2_Elo', 'P1_Surface_Elo', 'P2_Surface_Elo', 
        'Elo_Diff', 'Surface_Elo_Diff', 'H2H_Diff', 'Form_Diff', 'Surf_Form_Diff',
        'P1_H2H', 'P2_H2H', 'P1_Last10', 'P2_Last10'
    ]
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    # Predict
    probs = model.predict_proba(df)[0]
    p_loss = probs[0] # P2 Wins
    p_win = probs[1]  # P1 Wins
    
    print("\n" + "-"*40)
    print("PREDICTION RESULTS")
    print("-" * 40)
    
    winner = p1_name if p_win > p_loss else p2_name
    confidence = max(p_win, p_loss) * 100
    
    print(f"Predicted Winner: {winner}")
    print(f"Confidence:       {confidence:.2f}%")
    print(f"P1 Win Prob:      {p_win*100:.2f}%")
    print(f"P2 Win Prob:      {p_loss*100:.2f}%")
    
    # Betting Advice
    print("\n" + "-"*40)
    print("BETTING ADVICE (Strategy: Thresh 0.65, MinOdds 1.7)")
    print("-" * 40)
    
    ev_win = (p_win * p1_odds) - 1
    ev_loss = (p_loss * p2_odds) - 1
    
    bet_placed = False
    
    if ev_win > 0 and p_win > THRESHOLD and p1_odds >= MIN_ODDS:
        print(f"✅ BET ON: {p1_name}")
        print(f"   Odds: {p1_odds}")
        print(f"   EV:   {ev_win*100:.2f}%")
        bet_placed = True
    elif ev_loss > 0 and p_loss > THRESHOLD and p2_odds >= MIN_ODDS:
        print(f"✅ BET ON: {p2_name}")
        print(f"   Odds: {p2_odds}")
        print(f"   EV:   {ev_loss*100:.2f}%")
        bet_placed = True
    else:
        print("❌ NO BET")
        if ev_win > 0 or ev_loss > 0:
            print("   (Positive EV found, but confidence/odds criteria not met)")
        else:
            print("   (No Positive EV found)")

if __name__ == "__main__":
    while True:
        predict_match()
        cont = input("\nPredict another match? (y/n): ")
        if cont.lower() != 'y':
            break
