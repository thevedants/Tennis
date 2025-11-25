from data import train, test, val
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import argparse
import joblib

# Define features and target
features = ['AvgL', 'AvgW', 'Pointsdiff', 'LPts', 'WPts', 'LRank', 'WRank', 
            'Round_encoded', 'Series_encoded', 'Surface_Clay', 'Surface_Grass', 
            'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of']

target = 'Win'

# Prepare the data
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]
X_val = val[features]
y_val = val[target]

print(f"Training XGBoost on {len(X_train)} samples...")

# Initialize XGBoost
# Using scale_pos_weight to handle potential class imbalance if any, 
# though tennis matches are usually 50/50 in this dataset structure (swapped rows).
# However, since we swapped rows to make "Win" = 0 or 1, it should be balanced.
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

# Calibrate the model
# Isotonic regression usually works better with enough data (>1000 samples)
print("Calibrating model probabilities...")
calibrated_xgb = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
calibrated_xgb.fit(X_train, y_train)

print("Model trained and calibrated.")

# Feature importance (extracting from the base estimator inside the calibrated wrapper)
# We take the average feature importance across the CV folds or just fit a base model to see importance
xgb_for_importance = XGBClassifier(n_estimators=100, random_state=42)
xgb_for_importance.fit(X_train, y_train)
feature_importance = pd.DataFrame({'feature': features, 'importance': xgb_for_importance.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

def calculate_return(X, y_true, odds_w, odds_l, threshold=0.5, min_odds=0.0):
    probs = calibrated_xgb.predict_proba(X)
    total_bets = 0
    total_return = 0
    correct_predictions = 0

    for prob, true, odd_w, odd_l in zip(probs, y_true, odds_w, odds_l):
        p_loss = prob[0]
        p_win = prob[1]

        # Value Betting Logic
        ev_win = (p_win * odd_w) - 1
        ev_loss = (p_loss * odd_l) - 1
        
        if ev_win > 0 and p_win > threshold:
            pred = 1
            predicted_odds = odd_w
        elif ev_loss > 0 and p_loss > threshold:
            pred = 0
            predicted_odds = odd_l
        else:
            continue # No value found

        if predicted_odds < min_odds:
            continue

        total_bets += 1

        if pred == 1:
            if true == 1:
                total_return += odd_w
                correct_predictions += 1
            else:
                total_return -= 1 # Lose stake (assuming unit stake)
                # Note: In the original code, it was total_return -= 1. 
                # If odds are decimal, profit is odds - 1. If we sum odds, we subtract 1 per bet later?
                # Let's stick to the original logic:
                # If win: return += odds. If lose: return -= 1.
                # Net return = Total Return - Total Bets (Wait, this logic in original file was slightly ambiguous)
                # Original: net_return = total_return - total_bets
                # If I bet $1 on 2.0 odds and win, I get $2 back. Profit $1.
                # Code: total_return += 2.0. End: 2.0 - 1.0 = 1.0. Correct.
                # If I lose: total_return -= 1. End: -1.0 - 1.0 = -2.0. INCORRECT.
                # If I lose, I just lose my $1 stake. My "return" is 0.
                # The original code had: else: total_return -= 1.
                # AND THEN: net_return = total_return - total_bets.
                # If I lose: Return becomes -1. Net becomes -1 - 1 = -2. Double counting loss?
                
                # Let's FIX the return logic here to be standard.
                # We will track PROFIT directly.
                pass
        else:
            if true == 0:
                total_return += odd_l
                correct_predictions += 1
            else:
                total_return -= 1
    
    # Let's correct the logic to be simpler:
    # We will just sum the PnL.
    # If win: PnL += (Odds - 1)
    # If lose: PnL -= 1
    
    # Re-writing the loop for PnL calculation to be safe and correct
    pnl = 0
    actual_bets = 0
    correct = 0
    
    for prob, true, odd_w, odd_l in zip(probs, y_true, odds_w, odds_l):
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
            actual_bets += 1
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

    if actual_bets == 0:
        return {
            'total_bets': 0,
            'roi': 0,
            'net_return': 0,
            'accuracy': 0
        }

    roi = (pnl / actual_bets) * 100
    acc = (correct / actual_bets) * 100
    
    return {
        'total_bets': actual_bets,
        'net_return': pnl,
        'roi': roi,
        'accuracy': acc
    }

# Grid Search
print("\n" + "="*30)
print("STARTING GRID SEARCH (XGBoost + Calibration)")
print("="*30)

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
min_odds_list = [1.01, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0]

all_results = []

print(f"Testing {len(thresholds) * len(min_odds_list)} combinations on Validation Set...")

for thresh in thresholds:
    for min_od in min_odds_list:
        res = calculate_return(X_val, y_val, val['MaxW'], val['MaxL'], threshold=thresh, min_odds=min_od)
        
        if res['total_bets'] > 10:
            result_entry = {
                'threshold': thresh,
                'min_odds': min_od,
                'roi': res['roi'],
                'total_bets': res['total_bets'],
                'net_return': res['net_return'],
                'accuracy': res['accuracy']
            }
            all_results.append(result_entry)

all_results.sort(key=lambda x: x['roi'], reverse=True)
top_10 = all_results[:10]

print("\n" + "="*80)
print("TOP 10 STRATEGIES ON VALIDATION SET")
print("="*80)
print(f"{'Rank':<5} {'Thresh':<8} {'MinOdds':<8} {'Bets':<6} {'ROI':<8} {'Net Return':<12} {'Acc':<8}")
print("-" * 80)

for i, res in enumerate(top_10):
    print(f"{i+1:<5} {res['threshold']:<8} {res['min_odds']:<8} {res['total_bets']:<6} {res['roi']:>6.2f}% ${res['net_return']:>10.2f} {res['accuracy']:>6.2f}%")

print("\n" + "="*80)
print("TESTING TOP 10 STRATEGIES ON TEST SET")
print("="*80)
print(f"{'Rank':<5} {'Thresh':<8} {'MinOdds':<8} {'Bets':<6} {'ROI':<8} {'Net Return':<12} {'Acc':<8}")
print("-" * 80)

for i, res in enumerate(top_10):
    thresh = res['threshold']
    min_od = res['min_odds']
    
    test_res = calculate_return(X_test, y_test, test['MaxW'], test['MaxL'], threshold=thresh, min_odds=min_od)
    
    print(f"{i+1:<5} {thresh:<8} {min_od:<8} {test_res['total_bets']:<6} {test_res['roi']:>6.2f}% ${test_res['net_return']:>10.2f} {test_res['accuracy']:>6.2f}%")

# Save model
joblib.dump(calibrated_xgb, 'xgb_model_calibrated.joblib')
print("\nModel saved to xgb_model_calibrated.joblib")
