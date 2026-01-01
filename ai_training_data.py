# ai_training_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
file_path = "sales_data.csv"  # cleaned dataset from Project 2

df= pd.read_csv(file_path, encoding='cp1252')  # or encoding='latin1'


print("Dataset loaded for AI training:")
print(df.head())

# -------------------------------
# Step 2: Handle Missing Values (if any)
# -------------------------------
df.fillna(0, inplace=True)  # safe fallback

# -------------------------------
# Step 3: Encode Categorical Variables
# -------------------------------
categorical_cols = ['PRODUCTLINE', 'COUNTRY', 'DEALSIZE']
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # save encoders for future use

print("\nCategorical columns encoded.")

# -------------------------------
# Step 4: Feature Scaling
# -------------------------------
numeric_cols = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Numeric features scaled using MinMaxScaler.")

# -------------------------------
# Step 5: Create New Features
# -------------------------------
# Example: Month, Year already available, create Revenue per Quantity
df['RevenuePerUnit'] = df['SALES'] / (df['QUANTITYORDERED'] + 1e-5)  # avoid division by zero

# -------------------------------
# Step 6: Split Data for ML
# -------------------------------
X = df[['QUANTITYORDERED','PRICEEACH','MSRP','PRODUCTLINE','COUNTRY','DEALSIZE','RevenuePerUnit']]
y = df['SALES']  # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData split into train and test sets:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# -------------------------------
# Step 7: Save Prepared Data
# -------------------------------
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Prepared AI training data saved as CSV files.")
