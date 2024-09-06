from data import exported_df
ret = 0
for index, row in exported_df.iterrows():
    if row['MaxW'] > row['MaxL']:
        if row['Win'] == 1:
            ret += row['B365W']  - 1 # Subtract 1 to account for the initial bet
        else:
            ret -= 1
    else:
        if row['Win'] == 0:
            ret += row['B365L']   - 1 # Subtract 1 to account for the initial bet
        else:
            ret -= 1

print(f"Final return: ${ret:.2f}")
