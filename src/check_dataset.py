import pandas as pd
import os

file_path = "../data/raw/vehicle_tracks_000.csv"

print("================================")
print(" TRAJECTORY DATASET CHECK")
print("================================")

if not os.path.exists(file_path):
    print("\n❌ Dataset nahi mili!")
    print("Expected location:")
    print(file_path)
    exit()

print("\n✅ Dataset mil gayi!")

df = pd.read_csv(file_path)

print("\nDataset Size:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)