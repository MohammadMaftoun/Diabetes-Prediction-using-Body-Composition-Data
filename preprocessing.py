"""Preprocessing for Diabetes Body Composition data"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ctgan import CTGAN

# Load data
data = pd.read_excel("Body Composition and Diabetes(1).xlsx")

# Initial data exploration
def summary(data):
    print(f'data shape: {data.shape}')
    summ = pd.DataFrame(columns=['dtype', 'missing', 'missing[%]', 'unique', 'min', 'max', 'median', 'std', 'outliers', 'lower_bound', 'upper_bound'])
    for col in data.columns:
        summ.loc[col, 'dtype'] = data[col].dtype
        summ.loc[col, 'missing'] = data[col].isnull().sum()
        summ.loc[col, 'missing[%]'] = data[col].isnull().sum() / len(data) * 100
        summ.loc[col, 'unique'] = data[col].nunique()
        summ.loc[col, 'min'] = data[col].min()
        summ.loc[col, 'max'] = data[col].max()
        summ.loc[col, 'median'] = data[col].median()
        summ.loc[col, 'std'] = data[col].std()
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        summ.loc[col, 'outliers'] = outliers.count()
        summ.loc[col, 'lower_bound'] = lower_bound
        summ.loc[col, 'upper_bound'] = upper_bound
    return summ

# Display initial data info
print(data.head())
print(data.info())
print(data.describe())
print(data.corr())
print(summary(data))

# Data cleaning
df = data.dropna(subset=['HasDiabetes'])

# Convert to numeric and handle missing values
data['GenderID'] = pd.to_numeric(data['GenderID'], errors='coerce').fillna(data['GenderID'].mode()[0]).astype(int)
data['HasDiabetes'] = pd.to_numeric(data['HasDiabetes'], errors='coerce').fillna(data['HasDiabetes'].mode()[0]).astype(int)
data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(data['Age'].mean()).astype(float)

print(summary(data))

# Scaling
exclude_columns = ['HasDiabetes']
excluded_data = data[exclude_columns]
scaling_data = data.drop(columns=exclude_columns)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(scaling_data)
scaled_df = pd.DataFrame(scaled_data, columns=scaling_data.columns)

# Check for duplicate columns
duplicate_columns = data.columns[data.columns.duplicated()].tolist()
if duplicate_columns:
    print(f"Duplicate column names found: {duplicate_columns}")

# Combine scaled and excluded data
final_df = pd.concat([scaled_df, excluded_data.reset_index(drop=True)], axis=1)

# Synthetic data generation with CTGAN
cat_feature = ['GenderID', 'HasDiabetes']
ctgan = CTGAN(verbose=True)
# Uncomment to train CTGAN model
# ctgan.fit(final_df, cat_feature, epochs=200)
# ctgan.save('DiBGAN.pkl')

# Load pre-trained CTGAN model
loaded = CTGAN.load('DiBGAN.pkl')

# Generate synthetic samples
samples = loaded.sample(13001)
samples = samples[samples['HasDiabetes'] == 1]
ctgan_result_df = pd.concat([final_df, samples])

# Save preprocessed data
ctgan_result_df.to_csv('preprocessed_data.csv', index=False)
print(ctgan_result_df['HasDiabetes'].value_counts())
print(summary(ctgan_result_df))
