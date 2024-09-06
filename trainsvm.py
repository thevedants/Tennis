from data import train, test, val
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Define features and target
features = ['WRank', 'LRank', 'WPts', 'LPts', 'Pointsdiff', 'B365W', 'B365L', 'AvgW', 'AvgL', 'Series_encoded', 'Round_encoded', 'Best of', 'Surface_Clay', 'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Date', 'Info']
target = 'Win'

# Prepare the data
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate return based on model predictions
def calculate_return(X, y_true, odds_w, odds_l):
    X_scaled = scaler.transform(X)
    y_pred = svm_model.predict(X_scaled)
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
