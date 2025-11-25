# debug_model.py
from data import train, test, val
import numpy as np

print("="*70)
print("MODEL DEBUGGING")
print("="*70)

# Check basic stats
print("\n1. DATASET SIZES:")
print(f"Train: {len(train)}, Test: {len(test)}, Val: {len(val)}")

# Check if data actually loaded
print("\n2. SAMPLE DATA CHECK:")
print("Train head:")
print(train.head(3))
print("\nTest head:")
print(test.head(3))
print("\nVal head:")
print(val.head(3))

# Check for NaNs
print("\n3. NULL CHECK:")
print(f"Train nulls: {train.isnull().sum().sum()}")
print(f"Test nulls: {test.isnull().sum().sum()}")
print(f"Val nulls: {val.isnull().sum().sum()}")

if train.isnull().sum().sum() > 0:
    print("\nColumns with nulls in train:")
    print(train.isnull().sum()[train.isnull().sum() > 0])

# Check feature distributions
print("\n4. FEATURE STATS:")
features = ['WRank', 'LRank', 'AvgW', 'AvgL', 'WPts', 'LPts', 'Pointsdiff']
for feat in features:
    if feat in train.columns:
        print(f"\n{feat}:")
        print(f"  Train: mean={train[feat].mean():.3f}, std={train[feat].std():.3f}, min={train[feat].min():.3f}, max={train[feat].max():.3f}")
        print(f"  Test:  mean={test[feat].mean():.3f}, std={test[feat].std():.3f}, min={test[feat].min():.3f}, max={test[feat].max():.3f}")
        print(f"  Val:   mean={val[feat].mean():.3f}, std={val[feat].std():.3f}, min={val[feat].min():.3f}, max={val[feat].max():.3f}")

# Check target distribution
print("\n5. TARGET DISTRIBUTION:")
print(f"Train Win rate: {train['Win'].mean():.3f}")
print(f"Test Win rate: {test['Win'].mean():.3f}")
print(f"Val Win rate: {val['Win'].mean():.3f}")

# Check if baseline betting would work
print("\n6. BASELINE STRATEGIES:")

# Always bet on favorite (lower rank)
train_fav = (train['WRank'] < train['LRank']).mean()
test_fav = (test['WRank'] < test['LRank']).mean()
val_fav = (val['WRank'] < val['LRank']).mean()
print(f"Favorite wins: Train={train_fav:.3f}, Test={test_fav:.3f}, Val={val_fav:.3f}")

# Always bet on lower odds
train_odds = ((train['AvgW'] < train['AvgL']) & (train['Win'] == 1)).sum() / len(train)
test_odds = ((test['AvgW'] < test['AvgL']) & (test['Win'] == 1)).sum() / len(test)
val_odds = ((val['AvgW'] < val['AvgL']) & (val['Win'] == 1)).sum() / len(val)
print(f"Lower odds wins: Train={train_odds:.3f}, Test={test_odds:.3f}, Val={val_odds:.3f}")

# Check for data leakage - are there impossible values?
print("\n7. SANITY CHECKS:")
print(f"Any negative ranks? Train: {(train['WRank'] < 0).any()}, Test: {(test['WRank'] < 0).any()}, Val: {(val['WRank'] < 0).any()}")
print(f"Any odds < 1? Train: {(train['AvgW'] < 1).any()}, Test: {(test['AvgW'] < 1).any()}, Val: {(val['AvgW'] < 1).any()}")
print(f"Win only 0 or 1? Train: {set(train['Win'].unique())}, Test: {set(test['Win'].unique())}, Val: {set(val['Win'].unique())}")

# Check if MaxW/MaxL exist
print("\n8. ODDS COLUMNS AVAILABLE:")
odds_cols = [col for col in train.columns if 'Avg' in col or 'Max' in col or 'B365' in col]
print(f"Available odds columns: {odds_cols}")

print("\n9. ALL COLUMNS:")
print(train.columns.tolist())