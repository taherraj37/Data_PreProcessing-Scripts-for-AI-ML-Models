#Step 1:
# Data_PreProcessing-Scripts-for-AI-ML-Models
Python script for Data PreProcessing to use for AI/ML model development
    import pandas as pd
import numpy as np

# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Display the first few rows of the dummy dataset
print(df_dummy.head())

#Step 2:
def load_data(df):
    return df

def handle_missing_values(df):
    return df.fillna(df.mean())  # For numeric data, fill missing values with the mean

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]  # Remove rows with any outliers

def scale_data(df):
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)

#Step 3:

# Save the cleaned and preprocessed DataFrame to a CSV file
save_data(df_dummy, 'preprocessed_dummy_data.csv')

print('Preprocessing complete. Preprocessed data saved as preprocessed_dummy_data.csv')

print(df_dummy.isnull().sum())

#Step 4:
print(df_dummy.describe())

print(df_dummy.head())

print(df_dummy.columns)
