import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler


# ==========================================
# SETTINGS
# ==========================================

FILE_PATH = "../data/raw/vehicle_tracks_000.csv"

PAST_FRAMES = 20
FUTURE_FRAMES = 10

FEATURES = [
    "x",
    "y",
    "vx",
    "vy",
    "psi_rad"
]

TARGETS = [
    "x",
    "y"
]


# ==========================================
# LOAD DATASET
# ==========================================

print("========================================")
print(" TRAJECTORY PREPROCESSING")
print("========================================")

df = pd.read_csv(FILE_PATH)

print("\nDataset loaded!")
print("Rows:", len(df))


# ==========================================
# SORT DATA
# ==========================================

df = df.sort_values(
    ["track_id", "frame_id"]
).reset_index(drop=True)


# ==========================================
# CHECK BASIC INFORMATION
# ==========================================

print("\nTotal vehicles:", df["track_id"].nunique())

print("Past frames :", PAST_FRAMES)
print("Future frames:", FUTURE_FRAMES)


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

vehicle_ids = df["track_id"].unique()

np.random.seed(42)

np.random.shuffle(vehicle_ids)

total_vehicles = len(vehicle_ids)

train_end = int(total_vehicles * 0.70)
val_end = int(total_vehicles * 0.85)

train_ids = vehicle_ids[:train_end]
val_ids = vehicle_ids[train_end:val_end]
test_ids = vehicle_ids[val_end:]


print("\nVehicle Split:")
print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ==========================================
# SPLIT DATA
# ==========================================

train_df = df[df["track_id"].isin(train_ids)].copy()
val_df = df[df["track_id"].isin(val_ids)].copy()
test_df = df[df["track_id"].isin(test_ids)].copy()


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = MinMaxScaler()

scaler.fit(train_df[FEATURES])

train_df[FEATURES] = scaler.transform(
    train_df[FEATURES]
)

val_df[FEATURES] = scaler.transform(
    val_df[FEATURES]
)

test_df[FEATURES] = scaler.transform(
    test_df[FEATURES]
)


# ==========================================
# SEQUENCE CREATION FUNCTION
# ==========================================

def create_sequences(data):

    X = []
    Y = []

    for track_id, vehicle in data.groupby("track_id"):

        vehicle = vehicle.sort_values("frame_id")

        feature_values = vehicle[FEATURES].values

        target_values = vehicle[TARGETS].values

        total_length = len(vehicle)

        required_length = PAST_FRAMES + FUTURE_FRAMES

        if total_length < required_length:
            continue

        for i in range(
            total_length - required_length + 1
        ):

            past_data = feature_values[
                i : i + PAST_FRAMES
            ]

            future_data = target_values[
                i + PAST_FRAMES :
                i + PAST_FRAMES + FUTURE_FRAMES
            ]

            X.append(past_data)
            Y.append(future_data)

    return np.array(X), np.array(Y)


# ==========================================
# CREATE SEQUENCES
# ==========================================

print("\nCreating training sequences...")

X_train, Y_train = create_sequences(train_df)

print("Creating validation sequences...")

X_val, Y_val = create_sequences(val_df)

print("Creating testing sequences...")

X_test, Y_test = create_sequences(test_df)


# ==========================================
# DISPLAY SHAPES
# ==========================================

print("\n========================================")
print(" SEQUENCE SHAPES")
print("========================================")

print("\nX_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("\nX_val:", X_val.shape)
print("Y_val:", Y_val.shape)

print("\nX_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ==========================================
# CREATE PROCESSED FOLDER
# ==========================================

os.makedirs(
    "../data/processed",
    exist_ok=True
)


# ==========================================
# SAVE DATA
# ==========================================

np.savez_compressed(
    "../data/processed/trajectory_sequences.npz",

    X_train=X_train,
    Y_train=Y_train,

    X_val=X_val,
    Y_val=Y_val,

    X_test=X_test,
    Y_test=Y_test
)


# ==========================================
# SAVE SCALER
# ==========================================

import joblib

joblib.dump(
    scaler,
    "../data/processed/scaler.pkl"
)


print("\n========================================")
print(" PREPROCESSING COMPLETED")
print("========================================")

print("\nSaved files:")

print(
    "data/processed/trajectory_sequences.npz"
)

print(
    "data/processed/scaler.pkl"
)