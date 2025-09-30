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

# fixing class imbalance problem
import pandas as pd
from imblearn.over_sampling import SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from ctgan import CTGAN

def balance_final_df(final_df, method="svmsmote", cat_features=None, epochs=200, random_state=42):
    """
    Balance final_df directly (no manual X, y separation).
    Assumes target column is 'HasDiabetes'.

    Parameters:
    -----------
    final_df : pd.DataFrame
        Input dataframe with target column 'HasDiabetes'
    method : str
        One of ["svmsmote", "adasyn", "randomundersampler", "ctgan"]
    cat_features : list
        List of categorical features (needed for CTGAN)
    epochs : int
        Number of training epochs for CTGAN
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    balanced_df : pd.DataFrame
        Balanced dataframe including target column
    """

    if "HasDiabetes" not in final_df.columns:
        raise ValueError("final_df must contain 'HasDiabetes' as target column.")

    if method.lower() in ["svmsmote", "adasyn", "randomundersampler"]:
        # split internally just for sampler, but return as full df
        X = final_df.drop(columns=["HasDiabetes"])
        y = final_df["HasDiabetes"]

        if method.lower() == "svmsmote":
            sampler = SVMSMOTE(random_state=random_state)
        elif method.lower() == "adasyn":
            sampler = ADASYN(random_state=random_state)
        else:  # randomundersampler
            sampler = RandomUnderSampler(random_state=random_state)

        X_res, y_res = sampler.fit_resample(X, y)
        balanced_df = pd.concat([pd.DataFrame(X_res, columns=X.columns),
                                 pd.Series(y_res, name="HasDiabetes")], axis=1)

    elif method.lower() == "ctgan":
        if cat_features is None:
            raise ValueError("cat_features list must be provided for CTGAN.")

        df = final_df.copy()
        ctgan = CTGAN(verbose=True, epochs=epochs)
        ctgan.fit(df, cat_features)

        counts = df["HasDiabetes"].value_counts()
        max_class = counts.max()
        gen_samples = []

        for cls, count in counts.items():
            n_to_generate = max_class - count
            if n_to_generate > 0:
                gen_df = ctgan.sample(n_to_generate)
                gen_df = gen_df[gen_df["HasDiabetes"] == cls]
                gen_samples.append(gen_df)

        if gen_samples:
            gen_df = pd.concat(gen_samples, axis=0)
            balanced_df = pd.concat([df, gen_df], axis=0).reset_index(drop=True)
        else:
            balanced_df = df.copy()

    else:
        raise ValueError("Method must be one of ['svmsmote', 'adasyn', 'randomundersampler', 'ctgan']")

    return balanced_df


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

