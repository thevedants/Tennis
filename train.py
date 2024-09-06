from data import train, test, val

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
print(train.info())
    # Define the neural network architecture
class BinaryClassifier(nn.Module):
        def __init__(self, input_size):
            super(BinaryClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 8)
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

    # Prepare the data
X_train = train.drop('Win', axis=1).values
y_train = train['Win'].values

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=34, shuffle=True)

    # Initialize the model
input_size = X_train.shape[1]
model = BinaryClassifier(input_size)

    # Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
num_epochs = 100
for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed.")

    # Prepare the test data
X_test = test.drop('Win', axis=1).values
y_test = test['Win'].values

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    # Set the model to evaluation mode
model.eval()

    # Make predictions on the test set
with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_class = (y_pred > 0.5).float()

    # Calculate accuracy
correct = (y_pred_class == y_test_tensor).sum().item()
total = y_test_tensor.size(0)
accuracy = (correct / total) * 100

print(f"Test Accuracy: {accuracy:.2f}%") 

# Filter predictions based on confidence threshold
confidence_threshold_low = 0.3
confidence_threshold_high = 0.7
confident_mask = (y_pred < confidence_threshold_low) | (y_pred > confidence_threshold_high)

# Apply the mask to predictions and ground truth
y_pred_confident = y_pred[confident_mask]
y_test_confident = y_test_tensor[confident_mask]

# Convert predictions to class labels
y_pred_class_confident = (y_pred_confident > 0.5).float()

# Calculate accuracy for confident predictions
correct_confident = (y_pred_class_confident == y_test_confident).sum().item()
total_confident = y_test_confident.size(0)
accuracy_confident = (correct_confident / total_confident) * 100

print(f"Test Accuracy (confident predictions only): {accuracy_confident:.2f}%")
print(f"Number of confident predictions: {total_confident} out of {total}")
# Calculate expected profit for confident predictions with favorable odds
# Calculate expected profit for confident predictions with favorable odds
profit = 0
bet_amount = 100

# Combine test data with predictions
test_with_pred = test.copy()
test_with_pred['pred_prob'] = y_pred.numpy()

for idx, row in test_with_pred.iterrows():
    if row['pred_prob'] > 0.7 and row['MaxW'] > 1.7:
        if row['Win'] == 1:
            profit += bet_amount * (row['MaxW'] - 1)
        else:
            profit -= bet_amount


print(f"Expected profit: ${profit:.2f}")
print(f"Return on Investment (ROI): {(profit / (bet_amount * len(test_with_pred))) * 100:.2f}%")

