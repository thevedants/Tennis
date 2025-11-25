from data_enhanced import test
import joblib
import pandas as pd
import numpy as np

# Load Model
model = joblib.load('xgb_enhanced.joblib')

# Features
features = [
    'P1_Elo', 'P2_Elo', 'P1_Surface_Elo', 'P2_Surface_Elo', 
    'Elo_Diff', 'Surface_Elo_Diff', 
    'P1_Days_Since', 'P2_Days_Since', 
    'Rank_Diff', 'P1_Rank', 'P2_Rank', 'P1_Pts', 'P2_Pts', 
    'P1_H2H', 'P2_H2H', 'H2H_Diff',
    'P1_Last10', 'P2_Last10', 'Form_Diff',
    'P1_Last10_Surf', 'P2_Last10_Surf', 'Surf_Form_Diff',
    'Surface_Clay', 'Surface_Grass', 'Surface_Hard', 
    'Court_Indoor', 'Court_Outdoor', 'Best of', 
    'Round_encoded', 'Series_encoded'
]

# Prepare Data
# Ensure it's sorted by Date (it should be from data_enhanced, but let's be safe if Date column exists)
# data_enhanced.py keeps 'Date' in final_df but splits into train/test. 
# Let's check if 'Date' is in 'test'. data_enhanced.py: final_df['Date'] = ... then split. Yes.
if 'Date' in test.columns:
    test = test.sort_values('Date')

X_test = test[features]
y_test = test['Win']
odds_p1 = test['AvgW'] # P1 Odds
odds_p2 = test['AvgL'] # P2 Odds

# Get Probabilities
probs = model.predict_proba(X_test)

def run_staking_simulation(threshold, min_odds):
    bankroll = 100.0
    bet_amount = 1.0
    
    history = []
    max_bankroll = 100.0
    min_bankroll = 100.0
    
    total_bets = 0
    wins = 0
    
    # Iterate through matches chronologically
    for i in range(len(test)):
        prob = probs[i]
        true_outcome = y_test.iloc[i]
        odd_1 = odds_p1.iloc[i]
        odd_2 = odds_p2.iloc[i]
        
        p_loss = prob[0]
        p_win = prob[1]
        
        ev_win = (p_win * odd_1) - 1
        ev_loss = (p_loss * odd_2) - 1
        
        bet_placed = False
        bet_on_win = False
        
        # Strategy Logic
        if ev_win > 0 and p_win > threshold and odd_1 >= min_odds:
            bet_placed = True
            bet_on_win = True
            chosen_odds = odd_1
        elif ev_loss > 0 and p_loss > threshold and odd_2 >= min_odds:
            bet_placed = True
            bet_on_win = False
            chosen_odds = odd_2
            
        if bet_placed:
            total_bets += 1
            
            # Check if we can afford the bet
            if bet_amount > bankroll:
                bet_amount = bankroll # All in if bankroll is low
            
            won_bet = False
            if bet_on_win:
                if true_outcome == 1:
                    won_bet = True
            else:
                if true_outcome == 0:
                    won_bet = True
            
            if won_bet:
                profit = bet_amount * (chosen_odds - 1)
                bankroll += profit
                wins += 1
                # Strategy: If win, bet 1% of new bankroll
                bet_amount = bankroll / 100.0
            else:
                bankroll -= bet_amount
                # Strategy: If lose, reset to $1
                bet_amount = 1.0
            
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)
            history.append(bankroll)
            
            if bankroll <= 0:
                break

    roi = ((bankroll - 100) / 100) * 100 # ROI on initial bankroll
    accuracy = (wins / total_bets * 100) if total_bets > 0 else 0
    
    return {
        'Final Bankroll': bankroll,
        'Total Bets': total_bets,
        'Accuracy': accuracy,
        'Min Bankroll': min_bankroll,
        'Max Bankroll': max_bankroll,
        'Profit': bankroll - 100
    }

# Strategies to test (from previous best results)
strategies = [
    (0.65, 1.7), # The new Gold Standard
    (0.65, 1.6), # High Volume
    (0.6, 2.5),  # Sniper
    (0.6, 1.7),  # Balanced
    (0.6, 1.8)   # Balanced High Odds
]

print(f"{'Thresh':<8} {'MinOdds':<8} {'Bets':<6} {'Final $':<10} {'Profit':<10} {'Min $':<8} {'Max $':<8}")
print("-" * 70)

for thresh, min_od in strategies:
    res = run_staking_simulation(thresh, min_od)
    print(f"{thresh:<8} {min_od:<8} {res['Total Bets']:<6} ${res['Final Bankroll']:<9.2f} ${res['Profit']:<9.2f} ${res['Min Bankroll']:<7.2f} ${res['Max Bankroll']:<7.2f}")
