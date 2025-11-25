# val_analysis.py
from data import train, test, val
import pandas as pd
import numpy as np

print("="*70)
print("COMPREHENSIVE VALIDATION SET ANALYSIS")
print("="*70)

print("\n📊 BASIC STATS")
print(f"Total matches: {len(val)}")
print(f"Date range: Year {val['Date'].min()} to {val['Date'].max()} (2025 = year 12)")
print(f"Win rate: {val['Win'].mean():.3f} ({val['Win'].sum()} wins, {(~val['Win'].astype(bool)).sum()} losses)")

print("\n🎾 PLAYER RANKINGS")
print(f"Winner avg rank: {val['WRank'].mean():.1f} (median: {val['WRank'].median():.0f})")
print(f"Loser avg rank: {val['LRank'].mean():.1f} (median: {val['LRank'].median():.0f})")
print(f"Rank difference: {(val['LRank'] - val['WRank']).mean():.1f}")
print("\nRanking distribution:")
print(val[['WRank', 'LRank']].describe())

print("\n💰 ODDS ANALYSIS")
print(f"Average Winner odds: {val['AvgW'].mean():.2f} (median: {val['AvgW'].median():.2f})")
print(f"Average Loser odds: {val['AvgL'].mean():.2f} (median: {val['AvgL'].median():.2f})")
print(f"B365 Winner odds: {val['B365W'].mean():.2f}")
print(f"B365 Loser odds: {val['B365L'].mean():.2f}")
print("\nOdds distribution:")
print(val[['AvgW', 'AvgL', 'B365W', 'B365L']].describe())

print("\n🏆 TOURNAMENT BREAKDOWN")
print("Series distribution:")
series_dist = val['Series_encoded'].value_counts().sort_index()
series_map = {0: 'ATP250', 1: 'ATP500', 2: 'Grand Slam', 4: 'Masters 1000'}
for encoded_val, count in series_dist.items():
    series_name = series_map.get(encoded_val, f'Unknown({encoded_val*250})')
    print(f"  {series_name}: {count} matches ({count/len(val)*100:.1f}%)")

print("\nRound distribution:")
round_dist = val['Round_encoded'].value_counts().sort_index()
round_map = {0: '1st/RR', 1: '2nd', 2: '3rd', 3: '4th', 4: 'QF', 5: 'SF', 6: 'Final'}
for encoded_val, count in round_dist.items():
    round_name = round_map.get(encoded_val, f'Unknown({encoded_val})')
    print(f"  {round_name}: {count} matches ({count/len(val)*100:.1f}%)")

print("\n🎯 SURFACE & COURT")
print("Surface:")
print(f"  Clay: {val['Surface_Clay'].sum()} ({val['Surface_Clay'].mean()*100:.1f}%)")
print(f"  Grass: {val['Surface_Grass'].sum()} ({val['Surface_Grass'].mean()*100:.1f}%)")
print(f"  Hard: {val['Surface_Hard'].sum()} ({val['Surface_Hard'].mean()*100:.1f}%)")
print("\nCourt:")
print(f"  Indoor: {val['Court_Indoor'].sum()} ({val['Court_Indoor'].mean()*100:.1f}%)")
print(f"  Outdoor: {val['Court_Outdoor'].sum()} ({val['Court_Outdoor'].mean()*100:.1f}%)")

print("\n📈 POINTS ANALYSIS")
print(f"Winner points avg: {val['WPts'].mean():.3f} (normalized)")
print(f"Loser points avg: {val['LPts'].mean():.3f} (normalized)")
print(f"Points diff avg: {val['Pointsdiff'].mean():.3f}")
print(f"Points diff std: {val['Pointsdiff'].std():.3f}")

print("\n⚠️  UPSET POTENTIAL (Info column)")
print(f"Matches with upset potential (Info > 0): {(val['Info'] > 0).sum()}/{len(val)} ({(val['Info'] > 0).mean()*100:.1f}%)")
print(f"Average Info value: {val['Info'].mean():.2f}")
print(f"Max Info value: {val['Info'].max():.2f}")
upset_matches = val[val['Info'] > 0]
if len(upset_matches) > 0:
    print(f"When Info > 0, avg odds: {upset_matches['Info'].mean():.2f}")

print("\n🔥 COMPETITIVE MATCHES")
close_matches = val[abs(val['WRank'] - val['LRank']) < 10]
print(f"Close matches (rank diff < 10): {len(close_matches)}/{len(val)} ({len(close_matches)/len(val)*100:.1f}%)")
print(f"  Win rate in close matches: {close_matches['Win'].mean():.3f}")

favorites_won = val[val['WRank'] < val['LRank']]
print(f"\nFavorite won: {len(favorites_won)}/{len(val)} ({len(favorites_won)/len(val)*100:.1f}%)")
print(f"  Avg odds when favorite won: {favorites_won['AvgW'].mean():.2f}")

underdogs_won = val[val['WRank'] > val['LRank']]
print(f"Underdog won: {len(underdogs_won)}/{len(val)} ({len(underdogs_won)/len(val)*100:.1f}%)")
if len(underdogs_won) > 0:
    print(f"  Avg odds when underdog won: {underdogs_won['AvgW'].mean():.2f}")

print("\n🎲 BETTING MARKET EFFICIENCY")
# Check if odds correlate with actual outcomes
val_copy = val.copy()
val_copy['Favorite_Won'] = (val_copy['AvgW'] < val_copy['AvgL']) & (val_copy['Win'] == 1)
val_copy['Underdog_Won'] = (val_copy['AvgW'] > val_copy['AvgL']) & (val_copy['Win'] == 1)
print(f"Bookmaker favorite won: {val_copy['Favorite_Won'].sum()}/{len(val)} ({val_copy['Favorite_Won'].mean()*100:.1f}%)")
print(f"Bookmaker underdog won: {val_copy['Underdog_Won'].sum()}/{len(val)} ({val_copy['Underdog_Won'].mean()*100:.1f}%)")

print("\n❌ MISSING VALUES BREAKDOWN")
null_counts = val.isnull().sum()
if null_counts.sum() > 0:
    print("Columns with missing values:")
    for col, count in null_counts[null_counts > 0].items():
        print(f"  {col}: {count} ({count/len(val)*100:.1f}%)")
else:
    print("No missing values!")

print("\n🎰 ODDS CATEGORIES")
low_odds = val[val['AvgW'] < 1.5]
mid_odds = val[(val['AvgW'] >= 1.5) & (val['AvgW'] < 2.5)]
high_odds = val[val['AvgW'] >= 2.5]
print(f"Heavy favorites (AvgW < 1.5): {len(low_odds)} ({len(low_odds)/len(val)*100:.1f}%)")
print(f"  Win rate: {low_odds['Win'].mean():.3f}")
print(f"Moderate favorites (1.5 <= AvgW < 2.5): {len(mid_odds)} ({len(mid_odds)/len(val)*100:.1f}%)")
print(f"  Win rate: {mid_odds['Win'].mean():.3f}")
print(f"Underdogs/Even (AvgW >= 2.5): {len(high_odds)} ({len(high_odds)/len(val)*100:.1f}%)")
print(f"  Win rate: {high_odds['Win'].mean():.3f}")

print("\n👥 UNIQUE PLAYERS")
unique_winners = val['Winner_ID'].nunique()
unique_losers = val['Loser_ID'].nunique()
unique_players = len(set(val['Winner_ID'].unique()) | set(val['Loser_ID'].unique()))
print(f"Unique winners: {unique_winners}")
print(f"Unique losers: {unique_losers}")
print(f"Total unique players: {unique_players}")

print("\n🏅 BEST OF FORMAT")
best_of_dist = val['Best of'].value_counts().sort_index()
for bo, count in best_of_dist.items():
    print(f"  Best of {int(bo)}: {count} matches ({count/len(val)*100:.1f}%)")

print("\n📉 COMPARISON TO TRAIN SET")
print(f"{'Metric':<20} {'Train':<15} {'Val':<15} {'Diff':<15}")
print("-"*70)
metrics = {
    'Win Rate': (train['Win'].mean(), val['Win'].mean()),
    'Avg WRank': (train['WRank'].mean(), val['WRank'].mean()),
    'Avg LRank': (train['LRank'].mean(), val['LRank'].mean()),
    'Avg AvgW': (train['AvgW'].mean(), val['AvgW'].mean()),
    'Avg AvgL': (train['AvgL'].mean(), val['AvgL'].mean()),
    'Upset Rate': ((train['Info'] > 0).mean(), (val['Info'] > 0).mean()),
}
for metric_name, (train_val, val_val) in metrics.items():
    diff = val_val - train_val
    print(f"{metric_name:<20} {train_val:<15.3f} {val_val:<15.3f} {diff:+.3f}")

print("\n🔍 SAMPLE ROWS (first 10)")
display_cols = ['WRank', 'LRank', 'AvgW', 'AvgL', 'Win', 'Info', 'Surface_Hard', 'Round_encoded']
print(val[display_cols].head(10).to_string(index=False))

print("\n" + "="*70)