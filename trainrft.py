from data import train, test, val
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("-m", "--minodds", type=float, default=0.0, help="Minimum odds required to place bet")
args = parser.parse_args()

threshold = args.threshold
min_odds = args.minodds

print(f"Using threshold: {threshold}")
print(f"Using minimum odds: {min_odds}")


# Define features and target
features = ['AvgL', 'AvgW', 'Pointsdiff', 'LPts', 'WPts', 'LRank', 'WRank', 'Round_encoded', 'Series_encoded', 'Surface_Clay', 'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Best of']

target = 'Win'

# Prepare the data
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance)

# Calculate return based on model predictions
def calculate_return(X, y_true, odds_w, odds_l, threshold=0.5, min_odds=0.0):
    probs = rf_model.predict_proba(X)
    total_bets = 0
    total_return = 0
    correct_predictions = 0

    for prob, true, odd_w, odd_l in zip(probs, y_true, odds_w, odds_l):
        p_loss = prob[0]
        p_win = prob[1]

        # Value Betting Logic
        # Calculate implied probability from odds (1 / odds)
        implied_prob_win = 1 / odd_w
        implied_prob_loss = 1 / odd_l
        
        # Calculate Expected Value (EV)
        # EV = (Probability * Odds) - 1
        ev_win = (p_win * odd_w) - 1
        ev_loss = (p_loss * odd_l) - 1
        
        # Only bet if there is positive expected value (and it meets our threshold confidence)
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
                total_return -= 1
        else:
            if true == 0:
                total_return += odd_l
                correct_predictions += 1
            else:
                total_return -= 1

    if total_bets == 0:
        return {
            'total_bets': 0,
            'total_return': 0,
            'net_return': 0,
            'roi': 0,
            'accuracy': 0
        }

    net_return = total_return - total_bets
    roi = (net_return / total_bets) * 100
    accuracy = correct_predictions / total_bets * 100

    return {
        'total_bets': total_bets,
        'total_return': total_return,
        'net_return': net_return,
        'roi': roi,
        'accuracy': accuracy
    }



# Grid Search for Best Betting Parameters
print("\n" + "="*30)
print("STARTING GRID SEARCH FOR ROI MAXIMIZATION")
print("="*30)

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
min_odds_list = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 10.0]

all_results = []

X_val = val[features]
y_val = val[target]

# We need to get probabilities for the validation set once to speed up the loop
val_probs = rf_model.predict_proba(X_val)

print(f"Testing {len(thresholds) * len(min_odds_list)} combinations...")

for thresh in thresholds:
    for min_od in min_odds_list:
        res = calculate_return(X_val, y_val, val['MaxW'], val['MaxL'], threshold=thresh, min_odds=min_od)
        
        # We only care about strategies that actually place bets
        if res['total_bets'] > 10: # Minimum 10 bets to be considered
            result_entry = {
                'threshold': thresh,
                'min_odds': min_od,
                'roi': res['roi'],
                'total_bets': res['total_bets'],
                'net_return': res['net_return'],
                'accuracy': res['accuracy']
            }
            all_results.append(result_entry)

# Sort by ROI descending
all_results.sort(key=lambda x: x['roi'], reverse=True)

# Get Top 10
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


# Get the last 50 rows of the validation set
last_50_val = X_val.tail(50)
last_50_y_val = y_val.tail(50)
last_50_odds_w = val['MaxW'].tail(50)
last_50_odds_l = val['MaxL'].tail(50)

# Get predictions for the last 50 rows
last_50_predictions = rf_model.predict(last_50_val)
last_50_probabilities = rf_model.predict_proba(last_50_val)  # ADD THIS!

# Calculate returns for the last 50 rows
last_50_returns = []
for pred, true, odd_w, odd_l in zip(last_50_predictions, last_50_y_val, last_50_odds_w, last_50_odds_l):
    if pred == 1:  # Model predicts a win
        bet = "Win"
        if true == 1:
            return_value = odd_w - 1
        else:
            return_value = -1
    else:  # Model predicts a loss
        bet = "Loss"
        if true == 0:
            return_value = odd_l - 1
        else:
            return_value = -1
    last_50_returns.append(return_value)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'True Outcome': last_50_y_val,
    'Predicted Outcome': last_50_predictions,
    'Confidence Win': [prob[1] for prob in last_50_probabilities],  # ADD THIS!
    'Confidence Loss': [prob[0] for prob in last_50_probabilities],  # ADD THIS!
    'Bet': ['Win' if pred == 1 else 'Loss' for pred in last_50_predictions],
    'Return': last_50_returns
})

# Print the results
print("\nLast 50 rows of validation set with predictions and returns:")
print(results_df)


# Save the trained model
import joblib

# Save the model
joblib.dump(rf_model, 'rf_model.joblib')

# Create a class to encapsulate the model and prediction logic
class RandomForestModel:
    def __init__(self):
        self.model = joblib.load('rf_model.joblib')
        
    def predict(self, features):
        # Make sure features are in the correct format
        if isinstance(features, pd.DataFrame):
            # Ensure the DataFrame has the correct columns
            required_columns = X_val.columns.tolist()
            for col in required_columns:
                if col not in features.columns:
                    features[col] = 0  # Add missing columns with default value
            
            # Reorder columns to match training data
            features = features[required_columns]
        else:
            raise ValueError("Input must be a pandas DataFrame")
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Convert prediction to a more interpretable format
        return "Win" if prediction[0] == 1 else "Loss"

print("Model saved and RandomForestModel class created for web application use.")



