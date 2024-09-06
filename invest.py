from data import train, test, val
import numpy as np

# ... existing code ...

def betting_strategy(dataset, bet_on_favorite=True, odds_threshold=1.7, bet_amount=1):
    total_bets = 0
    total_return = 0

    for _, match in dataset.iterrows():
        if bet_on_favorite:
            if match['AvgW'] < match['AvgL']:
                odds = match['AvgW']
                won = match['Win'] == 1
                max_odds = match['MaxW']
            else:
                odds = match['AvgL']
                won = match['Win'] == 0
                max_odds = match['MaxL']
            
            if odds > odds_threshold:
                total_bets += bet_amount
                if won:
                    total_return += bet_amount * max_odds
        else:
            if match['AvgL'] > match['AvgW']:
                odds = match['AvgL']
                won = match['Win'] == 0
                max_odds = match['MaxL']
            else:
                odds = match['AvgW']
                won = match['Win'] == 1
                max_odds = match['MaxW']
            
            if odds < odds_threshold:  # Changed from > to <
                total_bets += bet_amount
                if won:
                    total_return += bet_amount * max_odds  # Use max_odds instead of odds

    net_return = total_return - total_bets
    roi = (net_return / total_bets) * 100 if total_bets > 0 else 0

    return {
        'total_bets': total_bets,
        'total_return': total_return,
        'net_return': net_return,
        'roi': roi
    }

# ... existing code ...

# For underdog betting, we'll now use lower thresholds
thresholds = np.arange(1.5, 3.05, 0.05)  # Adjusted range for underdog betting
underdog_results = []

for threshold in thresholds:
    result = betting_strategy(val, bet_on_favorite=False, odds_threshold=threshold, bet_amount=1)
    underdog_results.append({
        'threshold': threshold,
        'roi': result['roi'],
        'net_return': result['net_return'],
        'total_bets': result['total_bets']
    })

# Find the threshold with the maximum net return for underdog betting
max_underdog_result = max(underdog_results, key=lambda x: x['net_return'])

print("\nUnderdog Betting Results:")
print(f"Optimal odds threshold: {max_underdog_result['threshold']:.2f}")
print(f"Maximum ROI: {max_underdog_result['roi']:.2f}%")
print(f"Net return: ${max_underdog_result['net_return']:.2f}")
print(f"Total bets: {max_underdog_result['total_bets']}")

thresholds = np.arange(1.0, 2.05, 0.05)
results = []

for threshold in thresholds:
    result = betting_strategy(val, bet_on_favorite=True, odds_threshold=threshold, bet_amount=1)
    results.append({
        'threshold': threshold,
        'roi': result['roi'],
        'net_return': result['net_return'],
        'total_bets': result['total_bets']
    })

# Find the best combination of favorite and underdog thresholds
best_combination = {'favorite_threshold': 0, 'underdog_threshold': 0, 'net_return': float('-inf')}

for fav_threshold in thresholds:
    fav_result = betting_strategy(val, bet_on_favorite=True, odds_threshold=fav_threshold, bet_amount=1)
    
    for underdog_threshold in np.arange(1.5, 3.05, 0.05):
        underdog_result = betting_strategy(val, bet_on_favorite=False, odds_threshold=underdog_threshold, bet_amount=1)
        
        combined_net_return = fav_result['net_return'] + underdog_result['net_return']
        
        if combined_net_return > best_combination['net_return']:
            best_combination = {
                'favorite_threshold': fav_threshold,
                'underdog_threshold': underdog_threshold,
                'net_return': combined_net_return
            }

print("\nBest Combination of Thresholds:")
print(f"Favorite odds threshold: {best_combination['favorite_threshold']:.2f}")
print(f"Underdog odds threshold: {best_combination['underdog_threshold']:.2f}")
print(f"Combined net return: ${best_combination['net_return']:.2f}")

# Betting strategy based on better rank
def bet_on_better_rank(data, bet_amount=1):
    net_return = 0
    total_bets = 0

    for _, row in data.iterrows():
        if row['WRank'] > row['LRank']:  # Lower rank is better
            if row['Win'] == 1:
                net_return += (row['MaxW'] - 1) * bet_amount
            else:
                net_return -= bet_amount
        else:
            if row['Win'] == 0:
                net_return += (row['MaxL'] - 1) * bet_amount
            else:
                net_return -= bet_amount
        total_bets += 1

    roi = (net_return / (total_bets * bet_amount)) * 100 if total_bets > 0 else 0

    return {
        'net_return': net_return,
        'total_bets': total_bets,
        'roi': roi
    }

# Apply the better rank betting strategy
better_rank_result = bet_on_better_rank(test)

print("\nBetter Rank Betting Results:")
print(f"Net return: ${better_rank_result['net_return']:.2f}")
print(f"Total bets: {better_rank_result['total_bets']}")
print(f"ROI: {better_rank_result['roi']:.2f}%")
