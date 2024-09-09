import joblib

# Load the scaler from the file
scaler = joblib.load('min_max_scaler.pkl')
import pandas as pd
import numpy as np

# Create a sample DataFrame with the specified columns
df = pd.DataFrame({
    'WPts': np.random.randint(0, 10000, 100),
    'LPts': np.random.randint(0, 10000, 100),
})

# Calculate Pointsdiff
df['Pointsdiff'] = df['WPts'] - df['LPts']

# Apply the scaler to the DataFrame
columns_to_scale = ['WPts', 'LPts', 'Pointsdiff']
df[columns_to_scale] = scaler.transform(df[columns_to_scale])

print(df.head())
