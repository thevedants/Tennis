from data_enhanced import train, test, val
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import joblib

# Define features and target
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

# Prepare the data
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]
X_val = val[features]
y_val = val[target]

print(f"Training Enhanced XGBoost on {len(X_train)} samples with {len(features)} features...")

# Initialize XGBoost
xgb_base = XGBClassifier(
    n_estimators=1000, # Increased for more features
    learning_rate=0.03, # Slower learning
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# Calibrate
print("Calibrating model probabilities...")
calibrated_xgb = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
calibrated_xgb.fit(X_train, y_train)

print("Model trained and calibrated.")

# Feature Importance
xgb_for_importance = XGBClassifier(n_estimators=100, random_state=42)
xgb_for_importance.fit(X_train, y_train)
feature_importance = pd.DataFrame({'feature': features, 'importance': xgb_for_importance.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

def calculate_return(X, y_true, odds_p1, odds_p2, threshold=0.5, min_odds=0.0):
    probs = calibrated_xgb.predict_proba(X)
    
    total_bets = 0
    pnl = 0
    correct = 0
    
    for prob, true, odd_1, odd_2 in zip(probs, y_true, odds_p1, odds_p2):
        p_loss = prob[0] # Prob of 0 (P2 wins)
        p_win = prob[1]  # Prob of 1 (P1 wins)
        
        # Value Betting Logic
        ev_win = (p_win * odd_1) - 1
        ev_loss = (p_loss * odd_2) - 1
        
        bet_placed = False
        bet_on_win = False # True if betting on P1
        
        if ev_win > 0 and p_win > threshold and odd_1 >= min_odds:
            bet_placed = True
            bet_on_win = True
        elif ev_loss > 0 and p_loss > threshold and odd_2 >= min_odds:
            bet_placed = True
            bet_on_win = False
            
        if bet_placed:
            total_bets += 1
            if bet_on_win: # Bet on P1
                if true == 1:
                    pnl += (odd_1 - 1)
                    correct += 1
                else:
                    pnl -= 1
            else: # Bet on P2
                if true == 0:
                    pnl += (odd_2 - 1)
                    correct += 1
                else:
                    pnl -= 1

    if total_bets == 0:
        return {'total_bets': 0, 'roi': 0, 'net_return': 0, 'accuracy': 0}

    roi = (pnl / total_bets) * 100
    acc = (correct / total_bets) * 100
    
    return {'total_bets': total_bets, 'net_return': pnl, 'roi': roi, 'accuracy': acc}

# Grid Search
print("\n" + "="*30)
print("STARTING GRID SEARCH (Enhanced Model)")
print("="*30)

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
min_odds_list = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5]

all_results = []

# Note: In data_enhanced, AvgW is P1_Odds and AvgL is P2_Odds
odds_p1 = test['AvgW']
odds_p2 = test['AvgL']

print(f"Testing {len(thresholds) * len(min_odds_list)} combinations on Test Set (2025)...")

for thresh in thresholds:
    for min_od in min_odds_list:
        res = calculate_return(X_test, y_test, odds_p1, odds_p2, threshold=thresh, min_odds=min_od)
        
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

# Filter for positive ROI
profitable_strategies = [res for res in all_results if res['roi'] > 0]

# Sort by Total Bets descending
profitable_strategies.sort(key=lambda x: x['total_bets'], reverse=True)

print("\n" + "="*80)
print("ALL PROFITABLE STRATEGIES (Sorted by Volume)")
print("="*80)
print(f"{'Rank':<5} {'Thresh':<8} {'MinOdds':<8} {'Bets':<6} {'ROI':<8} {'Net Return':<12} {'Acc':<8}")
print("-" * 80)

for i, res in enumerate(profitable_strategies):
    print(f"{i+1:<5} {res['threshold']:<8} {res['min_odds']:<8} {res['total_bets']:<6} {res['roi']:>6.2f}% ${res['net_return']:>10.2f} {res['accuracy']:>6.2f}%")

# Save model
joblib.dump(calibrated_xgb, 'xgb_enhanced.joblib')
print("\nModel saved to xgb_enhanced.joblib")
