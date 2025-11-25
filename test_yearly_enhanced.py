import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from collections import defaultdict
import joblib
import os

# Load the full dataset (we need to regenerate features for each split to be strictly correct, 
# but for speed we can reuse the feature-engineered dataframe if we are careful about data leakage.
# However, Elo calculation MUST be done chronologically. 
# To do this properly for CV, we should really recalculate Elo for each fold if we want to be 100% pure,
# BUT, Elo is an iterative metric. If we train on 2013-2023 and test on 2024, we need Elo up to 2024.
# The `data_enhanced.py` calculated Elo over the ENTIRE history chronologically.
# This is actually CORRECT for testing 2024, because the model would have had access to 2013-2023 history.
# The only potential leakage is if we test on 2015 but the Elo was influenced by 2016 data?
# No, Elo is calculated row-by-row sorted by date. So row N's Elo only depends on 0..N-1.
# So we can safely use the pre-calculated features from data_enhanced.py for this CV!
# We just need to slice by Year.

from data_enhanced import final_df

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

target = 'Win'

# Strategy Parameters (The "Gold Standard")
THRESHOLD = 0.65
MIN_ODDS = 1.7

print(f"Running Yearly CV with Strategy: Threshold={THRESHOLD}, Min Odds={MIN_ODDS}")
print("-" * 50)
print(f"{'Year':<6} {'Bets':<6} {'ROI':<8} {'Net Return':<12} {'Acc':<8}")
print("-" * 50)

years = sorted(final_df['Year'].unique())
total_profit = 0
total_bets = 0
correct_bets = 0

for test_year in years:
    # Skip first year (2013) as it has no history for Elo to build up? 
    # Actually Elo starts at 1500. It's fine.
    
    # Train on all years EXCEPT test_year
    # Wait, standard time-series CV should train on PAST years only?
    # If we train on 2024 and test on 2015, that's "cheating" (lookahead bias) because the model learns patterns from the future.
    # However, the user asked for "isolate one year for test and train on all others".
    # This is "Leave-One-Year-Out" Cross-Validation. It's a standard robustness check, even if not strictly time-series valid.
    # It answers: "If I had this model structure and data from other years, how would it perform on this year?"
    
    train_mask = final_df['Year'] != test_year
    test_mask = final_df['Year'] == test_year
    
    train_data = final_df[train_mask]
    test_data = final_df[test_mask]
    
    if len(test_data) == 0:
        continue
        
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    odds_p1 = test_data['AvgW'] # Remember AvgW is P1_Odds
    odds_p2 = test_data['AvgL'] # Remember AvgL is P2_Odds
    
    # Train Model
    xgb = XGBClassifier(
        n_estimators=500, # Reduced slightly for speed in loop
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    calibrated = CalibratedClassifierCV(xgb, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    # Predict
    probs = calibrated.predict_proba(X_test)
    
    # Calculate Returns
    year_bets = 0
    year_pnl = 0
    year_correct = 0
    
    for i in range(len(test_data)):
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
        
        if ev_win > 0 and p_win > THRESHOLD and odd_1 >= MIN_ODDS:
            bet_placed = True
            bet_on_win = True
        elif ev_loss > 0 and p_loss > THRESHOLD and odd_2 >= MIN_ODDS:
            bet_placed = True
            bet_on_win = False
            
        if bet_placed:
            year_bets += 1
            if bet_on_win:
                if true_outcome == 1:
                    year_pnl += (odd_1 - 1)
                    year_correct += 1
                else:
                    year_pnl -= 1
            else:
                if true_outcome == 0:
                    year_pnl += (odd_2 - 1)
                    year_correct += 1
                else:
                    year_pnl -= 1
                    
    roi = (year_pnl / year_bets * 100) if year_bets > 0 else 0
    acc = (year_correct / year_bets * 100) if year_bets > 0 else 0
    
    print(f"{test_year:<6} {year_bets:<6} {roi:>6.2f}% $ {year_pnl:>10.2f}  {acc:>6.2f}%")
    
    total_bets += year_bets
    total_profit += year_pnl
    correct_bets += year_correct

print("-" * 50)
total_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
total_acc = (correct_bets / total_bets * 100) if total_bets > 0 else 0
print(f"TOTAL  {total_bets:<6} {total_roi:>6.2f}% $ {total_profit:>10.2f}  {total_acc:>6.2f}%")
