import pandas as pd
import os
import numpy as np
import math
import joblib
pd.options.mode.chained_assignment = None
'''Sports prediction and betting models in the
machine learning age: The case of tennis

Tennis Betting Strategies based on Neural Networks


Neural Networks and Betting Strategies for Tennis

Tennis betting:
can statistics beat bookmakers?'''
# Directory containing the .xlsx files
folder_path = '/Users/thevedantsingh/Desktop/Tennis'

# Initialize an empty list to store DataFrames
dataframes = []

# Iterate over each file in the directory
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        # Extract the year from the file name (assuming the file name is just the year)
        year = file_name.split('.')[0]
        
        # Read the Excel file into a DataFrame
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        
        # Add a column for the year
        df['Year'] = year
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dataframes, ignore_index=True)
# One-hot encode the 'Surface' column

surface_dummies = pd.get_dummies(final_df['Surface'], prefix='Surface')
final_df = pd.concat([final_df, surface_dummies], axis=1)

# One-hot encode the 'Court' column
court_dummies = pd.get_dummies(final_df['Court'], prefix='Court')
final_df = pd.concat([final_df, court_dummies], axis=1)

# Delete the original 'Surface' and 'Court' columns
final_df = final_df.drop(['Surface', 'Court'], axis=1)

# Encode the 'Series' column
series_mapping = {
    'ATP250': 250,
    'Grand Slam': 2000,
    'ATP500': 500,
    'Masters 1000': 1000,
    'Masters Cup': 1000
}

final_df['Series_encoded'] = final_df['Series'].map(series_mapping)



# Define a custom mapping for the 'Round' column
round_mapping = {
    '1st Round': 0,
    '2nd Round': 1,
    '3rd Round': 2,
    '4th Round': 3,
    'Quarterfinals': 4,
    'Semifinals': 5,
    'The Final': 6,
    'Final': 6,
    'Round Robin': 0
}

# Apply the mapping to create a new 'Round_encoded' column
final_df['Round_encoded'] = final_df['Round'].map(round_mapping)

# Handle any unexpected values in the 'Round' column
if final_df['Round_encoded'].isnull().any():
    print("Warning: Some values in the 'Round' column were not mapped.")
    print("Unique unmapped values:", final_df[final_df['Round_encoded'].isnull()]['Round'].unique())
    # You might want to fill NaN values with a default, e.g., -1
    final_df['Round_encoded'] = final_df['Round_encoded'].fillna(0)

# Convert to integer type
final_df['Round_encoded'] = final_df['Round_encoded'].astype(int)



# Remove rows where all of the odds columns are empty
odds_columns = ['B365W', 'B365L', 'PSW', 'PSL']
final_df = final_df.dropna(subset=odds_columns, how='all')


# Extract year from the 'Date' column
final_df['Date'] = pd.to_datetime(final_df['Date']).dt.year

# Ensure 'Date' column contains only the year
final_df['Date'] = final_df['Date'].astype(int)
# Create a dictionary to store unique IDs for each player
player_id_mapping = {}
current_id = 0

# Function to get or assign an ID for a player
def get_player_id(player):
    global current_id
    if player not in player_id_mapping:
        player_id_mapping[player] = current_id
        current_id += 1
    return player_id_mapping[player]

# Create new columns for Winner ID and Loser ID
final_df['Winner_ID'] = final_df['Winner'].apply(get_player_id)
final_df['Loser_ID'] = final_df['Loser'].apply(get_player_id)


# Drop columns W1, L1, W2, L2, W3, L3, W4, L4, W5, L5
columns_to_drop = ['W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5']
final_df = final_df.drop(columns=columns_to_drop)



# Drop Wsets and Lsets columns
columns_to_drop = ['Wsets', 'Lsets']
final_df = final_df.drop(columns=columns_to_drop)

# Fill other odds columns with AvgW and AvgL if they have null values
odds_columns = ['B365W', 'B365L', 'PSW', 'PSL', 'EXW', 'EXL', 'LBW', 'LBL', 'SJW','SJL']
for col in odds_columns:
    if col.endswith('W'):
        final_df[col].fillna(final_df['AvgW'], inplace=True)
    elif col.endswith('L'):
        final_df[col].fillna(final_df['AvgL'], inplace=True)



# Print the shape of final_df
# Fill 'WPts' and 'LPts' columns with their respective means where null

# Drop rows where AvgW is empty
final_df = final_df.dropna(subset=['AvgW'])
# Drop rows where WRank or LRank is empty
final_df = final_df.dropna(subset=['WRank', 'LRank'])

# Fill 'Best of' column with 3 where null
final_df['Best of'].fillna(3, inplace=True)






# Drop rows where Comment is not "Completed"
final_df = final_df[final_df['Comment'] == 'Completed']



# Drop specified columns
columns_to_drop = ['ATP', 'Location', 'Tournament', 'Series', 'Round', 'Winner', 'Loser', 'Comment']
final_df = final_df.drop(columns=columns_to_drop)
# Convert boolean columns to int
bool_columns = final_df.select_dtypes(include=['bool']).columns
for col in bool_columns:
    final_df[col] = final_df[col].astype(int)

# Drop the 'Year' column
final_df = final_df.drop(columns=['Year'])

# Create a new column 'Win' initialized with 1
final_df['Win'] = 1

# Identify rows where Winner_ID is greater than Loser_ID
mask = final_df['Winner_ID'] > final_df['Loser_ID']

# Switch values for identified rows
final_df.loc[mask, ['WPts', 'LPts']] = final_df.loc[mask, ['LPts', 'WPts']].values
final_df.loc[mask, ['WRank', 'LRank']] = final_df.loc[mask, ['LRank', 'WRank']].values
final_df.loc[mask, ['Winner_ID', 'Loser_ID']] = final_df.loc[mask, ['Loser_ID', 'Winner_ID']].values
final_df.loc[mask, ['B365W', 'B365L']] = final_df.loc[mask, ['B365L', 'B365W']].values
final_df.loc[mask, ['PSW', 'PSL']] = final_df.loc[mask, ['PSL', 'PSW']].values
final_df.loc[mask, ['MaxW', 'MaxL']] = final_df.loc[mask, ['MaxL', 'MaxW']].values
final_df.loc[mask, ['AvgW', 'AvgL']] = final_df.loc[mask, ['AvgL', 'AvgW']].values
final_df.loc[mask, ['EXW', 'EXL']] = final_df.loc[mask, ['EXL', 'EXW']].values
final_df.loc[mask, ['LBW','LBL']] = final_df.loc[mask, ['LBL','LBW']].values
final_df.loc[mask, ['SJW', 'SJL']] = final_df.loc[mask, ['SJL', 'SJW']].values
# Delete all the odds columns
odds_columns = ['PSW', 'PSL', 'EXW', 'EXL', 'LBW', 'LBL', 'SJW', 'SJL', 'MaxW', 'MaxL']
final_df = final_df.drop(columns=odds_columns)


final_df["Pointsdiff"] = (final_df["WPts"] - final_df["LPts"])
final_df['WPts'].fillna(final_df['WPts'].mean(), inplace=True)
final_df['LPts'].fillna(final_df['LPts'].mean(), inplace=True)
final_df['Pointsdiff'].fillna(final_df['Pointsdiff'].mean(), inplace=True)
# Set 'Win' to 0 for the switched rows
final_df.loc[mask, 'Win'] = 0

from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# List of columns to scale
columns_to_scale = ['WPts', 'LPts', 'Pointsdiff']

# Apply MinMaxScaler to the specified columns for train, validation, and test sets

final_df[columns_to_scale] = scaler.fit_transform(final_df[columns_to_scale])
joblib.dump(scaler, 'min_max_scaler.pkl')
# Normalize Series_encoded by dividing by 250

final_df['Series_encoded'] = final_df['Series_encoded'] / 250

# Create the Info column
final_df['Info'] = 0

# Determine the favorite player based on rank
favorite_is_winner = final_df['WRank'] < final_df['LRank']

# Set Info for cases where the winner is the favorite
final_df.loc[(favorite_is_winner) & (final_df['AvgW'] > 2), 'Info'] = final_df['AvgW']

# Set Info for cases where the loser is the favorite
final_df.loc[(~favorite_is_winner) & (final_df['AvgL'] > 2), 'Info'] = final_df['AvgL']
# Subtract 2013 from every value in the 'Date' column
final_df['Date'] = final_df['Date'] - 2013


# Ensure the 'Date' column remains as integer type
final_df['Date'] = final_df['Date'].astype(int)





# At the end of your data.py file, add:
if __name__ == "__main__":
    print(f"Shape of final_df: {final_df.shape}")
    # Print the info of final_df
    print("\nInfo of final_df:")
    print(final_df.info())
    print("Data collated successfully.")
    print(final_df.describe().transpose())
        # ... other print statements ...
    
else:
    # Split the data into train, test, and validation sets
    import numpy as np

    # Create a mask for 2024 data
    mask_2024 = final_df['Date'] == 11  # 2024 - 2013 = 11

    # Get 50% of 2024 data for validation
    val = final_df[mask_2024].sample(frac=0.2, random_state=42)

    # Remove validation data from the main dataset
    remaining_data = final_df[~final_df.index.isin(val.index)]

    # Split remaining data into train and test sets
    train_test_split = np.random.rand(len(remaining_data)) < 0.8
    train = remaining_data[train_test_split]
    test = remaining_data[~train_test_split]

    print(f"Train set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Validation set shape: {val.shape}")

    # Make these datasets available for import
    __all__ = ['train', 'test', 'val', 'final_df']




