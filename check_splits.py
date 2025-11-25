# check_splits.py
from data import train, test, val

print("Train dates:", sorted(train['Date'].unique()))
print("Test dates:", sorted(test['Date'].unique()))
print("Val dates:", sorted(val['Date'].unique()))

# Check for overlap
train_indices = set(train.index)
test_indices = set(test.index)
val_indices = set(val.index)

print(f"\nTrain-Test overlap: {len(train_indices & test_indices)} rows")
print(f"Train-Val overlap: {len(train_indices & val_indices)} rows")
print(f"Test-Val overlap: {len(test_indices & val_indices)} rows")

# Check if they sum to total
print(f"\nTotal rows: {len(train) + len(test) + len(val)}")

# Verify win rates make sense
print(f"\nBaseline (always predict favorite by rank):")
train_baseline = ((train['WRank'] < train['LRank']) == train['Win']).mean()
test_baseline = ((test['WRank'] < test['LRank']) == test['Win']).mean()
val_baseline = ((val['WRank'] < val['LRank']) == val['Win']).mean()

print(f"Train baseline accuracy: {train_baseline:.3f}")
print(f"Test baseline accuracy: {test_baseline:.3f}")
print(f"Val baseline accuracy: {val_baseline:.3f}")