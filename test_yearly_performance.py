from data import final_df
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration from the "Winner" strategy
THRESHOLD = 0.75
MIN_ODDS = 1.4

# Features and Target
features = ['AvgL', 'AvgW', 'Pointsdiff', 'LPts', 'WPts', 'LRank', 'WRank', 
            'Round_encoded', 'Series_encoded', 'Surface_Clay', 'Surface_Grass', 
            'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of']
target = 'Win'

def calculate_metrics(model, X_test, y_test, odds_w, odds_l, threshold, min_odds):
    probs = model.predict_proba(X_test)
    
    total_bets = 0
    pnl = 0
    correct = 0
    
    for prob, true, odd_w, odd_l in zip(probs, y_test, odds_w, odds_l):
        p_loss = prob[0]
        p_win = prob[1]
        
        ev_win = (p_win * odd_w) - 1
        ev_loss = (p_loss * odd_l) - 1
        
        bet_placed = False
        bet_on_win = False
        
        if ev_win > 0 and p_win > threshold and odd_w >= min_odds:
            bet_placed = True
            bet_on_win = True
        elif ev_loss > 0 and p_loss > threshold and odd_l >= min_odds:
            bet_placed = True
            bet_on_win = False
            
        if bet_placed:
            total_bets += 1
            if bet_on_win:
                if true == 1:
                    pnl += (odd_w - 1)
                    correct += 1
                else:
                    pnl -= 1
            else: # Bet on loss
                if true == 0:
                    pnl += (odd_l - 1)
                    correct += 1
                else:
                    pnl -= 1

    if total_bets == 0:
        return 0, 0, 0, 0

    roi = (pnl / total_bets) * 100
    acc = (correct / total_bets) * 100
    return total_bets, pnl, roi, acc

print(f"{'Year':<6} {'Bets':<6} {'ROI':<8} {'Net Return':<12} {'Acc':<8}")
print("-" * 50)

# Get all unique years
years = sorted(final_df['Date'].unique())

results = []

for test_year in years:
    # 1. Split Data
    # Test set is the current year
    test_mask = final_df['Date'] == test_year
    train_mask = ~test_mask
    
    X_train = final_df.loc[train_mask, features]
    y_train = final_df.loc[train_mask, target]
    
    X_test = final_df.loc[test_mask, features]
    y_test = final_df.loc[test_mask, target]
    
    # 2. Train Model (Retrain from scratch to avoid leakage)
    xgb_base = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Calibrate
    calibrated_xgb = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
    calibrated_xgb.fit(X_train, y_train)
    
    # 3. Evaluate
    odds_w = final_df.loc[test_mask, 'MaxW']
    odds_l = final_df.loc[test_mask, 'MaxL']
    
    bets, pnl, roi, acc = calculate_metrics(calibrated_xgb, X_test, y_test, odds_w, odds_l, THRESHOLD, MIN_ODDS)
    
    real_year = test_year + 2013
    print(f"{real_year:<6} {bets:<6} {roi:>6.2f}% ${pnl:>10.2f} {acc:>6.2f}%")
    
    results.append({
        'Year': real_year,
        'Bets': bets,
        'ROI': roi,
        'Net Return': pnl,
        'Accuracy': acc
    })

# Calculate averages
avg_roi = np.mean([r['ROI'] for r in results])
total_profit = sum([r['Net Return'] for r in results])
total_bets = sum([r['Bets'] for r in results])

print("-" * 50)
print(f"TOTAL PROFIT: ${total_profit:.2f}")
print(f"TOTAL BETS:   {total_bets}")
print(f"AVERAGE ROI:  {avg_roi:.2f}%")
