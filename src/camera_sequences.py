import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "calibrated_camera_trajectories.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "camera_trajectory_sequences.npz"
)

SCALER_FILE = os.path.join(
    OUTPUT_DIR,
    "camera_scaler.pkl"
)

# ============================================================
# SETTINGS
# ============================================================

PAST_FRAMES = 20
FUTURE_FRAMES = 10

FEATURES = [
    "world_x",
    "world_y",
    "vx",
    "vy",
    "speed"
]

TARGETS = [
    "world_x",
    "world_y"
]

# ============================================================
# LOAD
# ============================================================

print("\nLoading camera trajectory data...")

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["track_id", "frame_id"]
).reset_index(drop=True)

df = df.dropna(
    subset=FEATURES + TARGETS
)

print("Total records:", len(df))
print("Unique tracks:", df["track_id"].nunique())

# ============================================================
# CREATE ALL VALID SEQUENCES
# ============================================================

all_X = []
all_Y = []

print("\nCreating sequences...")

for track_id in df["track_id"].unique():

    track = df[
        df["track_id"] == track_id
    ].sort_values("frame_id")

    if len(track) < PAST_FRAMES + FUTURE_FRAMES:
        continue

    values = track[FEATURES].values
    targets = track[TARGETS].values

    for i in range(
        len(track) - PAST_FRAMES - FUTURE_FRAMES + 1
    ):

        past = values[
            i:i + PAST_FRAMES
        ]

        future = targets[
            i + PAST_FRAMES:
            i + PAST_FRAMES + FUTURE_FRAMES
        ]

        all_X.append(past)
        all_Y.append(future)

print("Total valid sequences:", len(all_X))

# ============================================================
# CONVERT TO ARRAY
# ============================================================

all_X = np.array(
    all_X,
    dtype=np.float32
)

all_Y = np.array(
    all_Y,
    dtype=np.float32
)

# ============================================================
# SHUFFLE SEQUENCES
# ============================================================

np.random.seed(42)

indices = np.random.permutation(
    len(all_X)
)

all_X = all_X[indices]
all_Y = all_Y[indices]

# ============================================================
# SEQUENCE-LEVEL SPLIT
# ============================================================

total = len(all_X)

train_end = int(total * 0.70)

val_end = int(total * 0.85)

X_train = all_X[:train_end]
Y_train = all_Y[:train_end]

X_val = all_X[train_end:val_end]
Y_val = all_Y[train_end:val_end]

X_test = all_X[val_end:]
Y_test = all_Y[val_end:]

# ============================================================
# SCALER
# ============================================================

print("\nFitting scaler...")

scaler = MinMaxScaler()

scaler.fit(
    X_train.reshape(-1, len(FEATURES))
)

X_train_scaled = scaler.transform(
    X_train.reshape(-1, len(FEATURES))
).reshape(X_train.shape)

X_val_scaled = scaler.transform(
    X_val.reshape(-1, len(FEATURES))
).reshape(X_val.shape)

X_test_scaled = scaler.transform(
    X_test.reshape(-1, len(FEATURES))
).reshape(X_test.shape)

joblib.dump(
    scaler,
    SCALER_FILE
)

# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    X_train=X_train_scaled,
    Y_train=Y_train,
    X_val=X_val_scaled,
    Y_val=Y_val,
    X_test=X_test_scaled,
    Y_test=Y_test
)

# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("CAMERA DATASET READY")
print("======================================")

print("X_train:", X_train_scaled.shape)
print("Y_train:", Y_train.shape)

print("X_val:", X_val_scaled.shape)
print("Y_val:", Y_val.shape)

print("X_test:", X_test_scaled.shape)
print("Y_test:", Y_test.shape)

print("\nScaler saved:")
print(SCALER_FILE)

print("\nDataset saved:")
print(OUTPUT_FILE)

print("\nSUCCESS")