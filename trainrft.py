from data import train, test, val
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Define features and target
features = ['AvgL', 'AvgW', 'Pointsdiff', 'LPts', 'WPts', 'LRank', 'WRank', 'B365L', 'B365W', 'Date', 'Round_encoded', 'Series_encoded', 'Info', 'Surface_Hard', 'Surface_Clay']

target = 'Win'

# Prepare the data
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
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
def calculate_return(X, y_true, odds_w, odds_l):
    y_pred = rf_model.predict(X)
    total_bets = len(y_pred)
    total_return = 0
    correct_predictions = 0

    for pred, true, odd_w, odd_l in zip(y_pred, y_true, odds_w, odds_l):
        if pred == 1:  # Model predicts a win
            if true == 1:  # Correct prediction
                total_return += odd_w
                correct_predictions += 1
            else:
                total_return -= 1
        else:  # Model predicts a loss
            if true == 0:  # Correct prediction
                total_return += odd_l
                correct_predictions += 1
            else:
                total_return -= 1

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

# Calculate return for test set
test_return = calculate_return(X_test, y_test, test['B365W'], test['B365L'])

print("\nBetting Results on Test Set:")
print(f"Total bets: {test_return['total_bets']}")
print(f"Total return: ${test_return['total_return']:.2f}")
print(f"Net return: ${test_return['net_return']:.2f}")
print(f"ROI: {test_return['roi']:.2f}%")
print(f"Prediction Accuracy: {test_return['accuracy']:.2f}%")


# Calculate return for validation set

X_val = val[features]
y_val = val[target]

val_return = calculate_return(X_val, y_val, val['B365W'], val['B365L'])

print("\nBetting Results on Validation Set:")
print(f"Total bets: {val_return['total_bets']}")
print(f"Total return: ${val_return['total_return']:.2f}")
print(f"Net return: ${val_return['net_return']:.2f}")
print(f"ROI: {val_return['roi']:.2f}%")
print(f"Prediction Accuracy: {val_return['accuracy']:.2f}%")


# Get the last 50 rows of the validation set
last_50_val = X_val.tail(50)
last_50_y_val = y_val.tail(50)
last_50_odds_w = val['B365W'].tail(50)
last_50_odds_l = val['B365L'].tail(50)

# Get predictions for the last 50 rows
last_50_predictions = rf_model.predict(last_50_val)

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



