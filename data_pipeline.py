# data_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 1. EXTRACT: Load data (Example CSV or dummy data for illustration)
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    # Generating dummy data for testing if file doesn't exist
    df = pd.DataFrame({
        'Age': [25, 30, 22, 35],
        'Salary': [50000, 60000, 45000, 80000],
        'Department': ['HR', 'IT', 'Finance', 'IT']
    })
    df.to_csv('data.csv', index=False)
    print("Sample data file 'data.csv' created for testing.")

print("\nOriginal Data:")
print(df)

# 1.1 Clean Data — Remove negative salaries
initial_len = len(df)
df = df[df['Salary'] >= 0]
removed_rows = initial_len - len(df)
if removed_rows > 0:
    print(f"\nRemoved {removed_rows} row(s) with negative salary.")

# 2. TRANSFORM: Preprocessing pipeline
numerical_features = ['Age', 'Salary']
categorical_features = ['Department']

# Preprocessing transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

# Full preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform the data
transformed_data = preprocessor.fit_transform(df)

# Get new column names
cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_columns = numerical_features + list(cat_columns)

# Convert to DataFrame
transformed_df = pd.DataFrame(transformed_data, columns=all_columns)

print("\nTransformed Data:")
print(transformed_df)

# 3. LOAD: Save the transformed data to a new CSV
transformed_df.to_csv('transformed_data.csv', index=False)
print("\nTransformed data saved to 'transformed_data.csv'")
