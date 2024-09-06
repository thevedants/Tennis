from data import train, test, val
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define features and target
features = ['WRank', 'LRank', 'WPts', 'LPts', 'Pointsdiff', 'B365W', 'B365L', 'AvgW', 'AvgL', 'Series_encoded', 'Round_encoded', 'Best of', 'Surface_Clay', 'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor', 'Date', 'Info']
target = 'Win'

# Prepare the data
X_train = train[features].values
y_train = train[target].values
X_test = test[features].values
y_test = test[target].values

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# Initialize the model
model = Net(len(features))

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on validation set
X_val = val[features]
y_val = val[target]

# Scale the validation features
X_val_scaled = scaler.transform(X_val)

# Convert to PyTorch tensor
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val.values)

# Set the model to evaluation mode
model.eval()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Make predictions
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_predictions = (val_outputs.squeeze() > 0.5).float()

# Convert predictions to numpy for sklearn metrics
val_predictions_np = val_predictions.numpy()
y_val_np = y_val.values

# Calculate accuracy
accuracy = accuracy_score(y_val_np, val_predictions_np)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_val_np, val_predictions_np)
print("\nConfusion Matrix:")
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_val_np, val_predictions_np)
print("\nClassification Report:")
print(class_report)

# Set the model back to training mode
model.train()
# Evaluate the model on test set
X_test = test[features]
y_test = test[target]

# Scale the test features
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensor
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values)

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs.squeeze() > 0.5).float()

# Convert predictions to numpy for sklearn metrics
test_predictions_np = test_predictions.numpy()
y_test_np = y_test.values

# Calculate accuracy
test_accuracy = accuracy_score(y_test_np, test_predictions_np)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Generate confusion matrix
test_conf_matrix = confusion_matrix(y_test_np, test_predictions_np)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)

# Generate classification report
test_class_report = classification_report(y_test_np, test_predictions_np)
print("\nTest Classification Report:")
print(test_class_report)

# Set the model back to training mode
model.train()

# Print the first 10 values of test_outputs
print("\nFirst 10 values of test_outputs:")
print(test_outputs[:10].squeeze().tolist())

# Calculate accuracy for high confidence predictions (>0.7 or <0.3) on test set
high_confidence_mask = (test_outputs.squeeze() > 0.7) | (test_outputs.squeeze() < 0.3)
high_confidence_predictions = test_predictions[high_confidence_mask]
high_confidence_true = y_test_tensor[high_confidence_mask]

high_confidence_accuracy = accuracy_score(high_confidence_true.numpy(), high_confidence_predictions.numpy())
high_confidence_count = high_confidence_mask.sum().item()
high_confidence_percentage = (high_confidence_count / len(test_outputs)) * 100

print(f"\nHigh Confidence Predictions (>0.7 or <0.3):")
print(f"Number of high confidence predictions: {high_confidence_count}")
print(f"Percentage of high confidence predictions: {high_confidence_percentage:.2f}%")
print(f"Accuracy of high confidence predictions: {high_confidence_accuracy:.4f}")

# Calculate return for high confidence predictions with favorable odds
def calculate_return(outputs, y_true, odds, avg_odds, threshold=0.7, avg_odds_threshold=1.7):
    total_bets = 0
    total_return = 0
    correct_predictions = 0

    for pred, true, odd, avg_odd in zip(outputs, y_true, odds, avg_odds):
        if pred > threshold and avg_odd > avg_odds_threshold:
            total_bets += 1
            if true == 1:  # Correct prediction
                total_return += odd
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
    accuracy = (correct_predictions / total_bets) * 100

    return {
        'total_bets': total_bets,
        'total_return': total_return,
        'net_return': net_return,
        'roi': roi,
        'accuracy': accuracy
    }

# Get the necessary data from the test set
test_outputs_np = test_outputs.squeeze().numpy()
test_odds = test['B365W'].values
test_avg_odds = test['AvgW'].values

# Calculate return for test set
test_return = calculate_return(test_outputs_np, y_test_np, test_odds, test_avg_odds)

print("\nBetting Results on Test Set (prob > 0.7 and AvgW > 1.7):")
print(f"Total bets: {test_return['total_bets']}")
print(f"Total return: ${test_return['total_return']:.2f}")
print(f"Net return: ${test_return['net_return']:.2f}")
print(f"ROI: {test_return['roi']:.2f}%")
print(f"Prediction Accuracy: {test_return['accuracy']:.2f}%")

# Calculate return for validation set
X_val_tensor = torch.FloatTensor(scaler.transform(val[features]))
with torch.no_grad():
    val_outputs = model(X_val_tensor)
val_outputs_np = val_outputs.squeeze().numpy()
val_y_np = val[target].values
val_odds = val['B365W'].values
val_avg_odds = val['AvgW'].values

val_return = calculate_return(val_outputs_np, val_y_np, val_odds, val_avg_odds)

print("\nBetting Results on Validation Set (prob > 0.7 and AvgW > 1.7):")
print(f"Total bets: {val_return['total_bets']}")
print(f"Total return: ${val_return['total_return']:.2f}")
print(f"Net return: ${val_return['net_return']:.2f}")
print(f"ROI: {val_return['roi']:.2f}%")
print(f"Prediction Accuracy: {val_return['accuracy']:.2f}%")

# Function to calculate return for underdog bets
def calculate_underdog_return(probabilities, y_true, odds_l, avg_odds_l):
    total_bets = 0
    total_return = 0
    correct_predictions = 0

    for prob, true, odd_l, avg_odd_l in zip(probabilities, y_true, odds_l, avg_odds_l):
        if prob < 0.3 and avg_odd_l > 1.7:
            total_bets += 1
            if true == 0:  # Underdog wins
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
    accuracy = (correct_predictions / total_bets) * 100

    return {
        'total_bets': total_bets,
        'total_return': total_return,
        'net_return': net_return,
        'roi': roi,
        'accuracy': accuracy
    }

# Calculate underdog return for test set
test_underdog_return = calculate_underdog_return(test_outputs_np, y_test_np, test['B365L'].values, test['AvgL'].values)

print("\nUnderdog Betting Results on Test Set (prob < 0.3 and AvgL > 1.7):")
print(f"Total bets: {test_underdog_return['total_bets']}")
print(f"Total return: ${test_underdog_return['total_return']:.2f}")
print(f"Net return: ${test_underdog_return['net_return']:.2f}")
print(f"ROI: {test_underdog_return['roi']:.2f}%")
print(f"Prediction Accuracy: {test_underdog_return['accuracy']:.2f}%")

# Calculate underdog return for validation set
val_underdog_return = calculate_underdog_return(val_outputs_np, val_y_np, val['B365L'].values, val['AvgL'].values)

print("\nUnderdog Betting Results on Validation Set (prob < 0.3 and AvgL > 1.7):")
print(f"Total bets: {val_underdog_return['total_bets']}")
print(f"Total return: ${val_underdog_return['total_return']:.2f}")
print(f"Net return: ${val_underdog_return['net_return']:.2f}")
print(f"ROI: {val_underdog_return['roi']:.2f}%")
print(f"Prediction Accuracy: {val_underdog_return['accuracy']:.2f}%")
