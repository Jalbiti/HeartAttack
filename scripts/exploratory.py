# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('echocardiogram.data', 'r') as file:
    lines = file.readlines()

# Process each line manually to handle variable columns
data = [line.strip().split(',') for line in lines]

# Known column names
columns = ['survival', 'still-alive', 'age-at-heart-attack', 'pericardial-effusion', 'fractional-shortening',
           'epss', 'lvdd', 'wall-motion-score', 'wall-motion-index', 'mult', 'name', 'group', 'alive-at-1', 'unknown']

# Create DataFrame
df = pd.DataFrame(data, columns=columns)
df.drop(columns=['unknown'], inplace=True)

# Check data types before conversion
print(df.dtypes)

# Specify the desired data types
# Use a dictionary to map column names to types
dtypes = {
    'survival': 'float',
    'still-alive': 'int',
    'age-at-heart-attack': 'float',
    'pericardial-effusion': 'int',
    'fractional-shortening': 'float',
    'epss': 'float',
    'lvdd': 'float',
    'wall-motion-score': 'float',
    'wall-motion-index': 'float',
    'mult': 'float',
    'name': 'string',
    'group': 'string',
    'alive-at-1': 'int'
}

# Convert the DataFrame and handle errors
for column, dtype in dtypes.items():
    if column in df.columns:
        # Convert and drop rows with errors
        df[column] = pd.to_numeric(df[column], errors='coerce') if dtype in ['int', 'float'] else df[column].astype(dtype)

# As we can observe, we get a lot of NaNs in the alive-at-1, so we can process the information to avoid it
# If a patient is not still alive and survival < 12, then we know that alive-at-1 is False
df.loc[(df['survival'] < 12) & (df['still-alive'] == 0), 'alive-at-1'] = 0

# If a patient is still alive and survival > 12, then we know that alive-at-1 is True
df.loc[(df['survival'] >= 12) & (df['still-alive'] == 1), 'alive-at-1'] = 1

# We actually should not remove NaN values as this removes many cases where:
# either survival > 12, still-alive = 0 and alive-at-1 is NaN
# or survival < 12, still-alive = 1 and alive-at-1 is NaN

# This cases can be used for testing (not for validating as we don't know the ground truth)

# Potentially remove patients without data or those that cannot be used for the prediction
#  (i.e. survived less than one year but still alive)

filtered_df = df[~((df['survival'] < 12) & (df['still-alive'] == 1))].copy()

# Drop rows with NaN values (indicating conversion errors)
df.dropna(subset=['survival'], inplace=True)
df.to_csv("input.csv", index=False, quoting=False)

filtered_df.dropna(subset=['survival'], inplace=True)

df_clean = df.drop(columns=['name', 'group', 'mult'])
plt.figure(figsize=(13, 12))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.savefig('corr_matrix.png')

sns.pairplot(df_clean, hue='alive-at-1', diag_kind='kde')
plt.title("Pair Plot of Features")
plt.savefig('dist_matrix.png')
