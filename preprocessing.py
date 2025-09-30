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

# Distribution comparison
def plot_distributions(real_data, synthetic_data, columns, n_cols=4):
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    axes = axes.ravel() if n_rows > 1 else [axes]

    for idx, col in enumerate(columns):
        if col in real_data.columns and col in synthetic_data.columns:
            if col in ['GenderID', 'HasDiabetes']:
                # Count plot for categorical features
                real_counts = real_data[col].value_counts().reset_index()
                real_counts['Dataset'] = 'Real'
                synth_counts = synthetic_data[col].value_counts().reset_index()
                synth_counts['Dataset'] = 'Synthetic'
                combined_counts = pd.concat([real_counts, synth_counts])
                combined_counts.columns = [col, 'Count', 'Dataset']

                sns.barplot(data=combined_counts, x=col, y='Count', hue='Dataset',
                           palette={'Real': 'blue', 'Synthetic': 'red'}, ax=axes[idx])
                axes[idx].set_title(f'Count Plot of {col}')
            else:
                # Histogram and KDE for numerical features
                sns.histplot(data=real_data, x=col, color='blue', alpha=0.4, label='Real', ax=axes[idx])
                sns.histplot(data=synthetic_data, x=col, color='red', alpha=0.4, label='Synthetic', ax=axes[idx])
                sns.kdeplot(data=real_data, x=col, color='blue', label='Real KDE', linestyle='--', ax=axes[idx])
                sns.kdeplot(data=synthetic_data, x=col, color='red', label='Synthetic KDE', linestyle='--', ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')

            axes[idx].legend()

    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

# Select all features for comparison
key_features = list(final_df.columns)
plot_distributions(final_df, samples, key_features)

# Print value counts
print("\nHasDiabetes value counts in final dataset:")
print(ctgan_result_df['HasDiabetes'].value_counts())
