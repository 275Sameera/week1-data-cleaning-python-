
# Week 1: Data Acquisition, Cleaning, and Preprocessing

import pandas as pd
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()

# Give the target column a readable species name
df = df.rename(columns={"target": "species_code"})
df["species"] = df["species_code"].map(
    dict(enumerate(iris.target_names))
)
df = df.drop(columns=["species_code"])

# 2. Explore the dataset
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

# 3. Check missing values and duplicates
print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 4. Remove duplicate rows
clean = df.drop_duplicates().copy()

# 5. Handle missing numerical values
numeric_cols = clean.select_dtypes(
    include="number"
).columns

for col in numeric_cols:
    if clean[col].isna().any():
        clean[col] = clean[col].fillna(
            clean[col].median()
        )

# Handle missing categorical values
categorical_cols = clean.select_dtypes(
    exclude="number"
).columns

for col in categorical_cols:
    if clean[col].isna().any():
        mode_values = clean[col].mode()
        if not mode_values.empty:
            clean[col] = clean[col].fillna(
                mode_values.iloc[0]
            )

# 6. Identify potential outliers using IQR
for col in numeric_cols:
    q1 = clean[col].quantile(0.25)
    q3 = clean[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (clean[col] < lower) |
        (clean[col] > upper)
    ).sum()

    print(f"Potential outliers in {col}: {count}")

# 7. Validate the cleaned data
print("\nCleaned dataset shape:", clean.shape)
print("\nMissing values after cleaning:")
print(clean.isna().sum())

# 8. Export cleaned dataset
clean.to_csv("iris_cleaned.csv", index=False)
print("\nSaved cleaned data to iris_cleaned.csv")
