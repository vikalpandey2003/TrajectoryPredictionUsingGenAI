import cv2
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
from tensorflow.keras.models import load_model

print("========================================")
print(" CAMERA + LSTM TRAJECTORY PREDICTION")
print("========================================")

MODEL_PATH = "../models/lstm_trajectory.keras"
SCALER_PATH = "../data/processed/scaler.pkl"

PAST_FRAMES = 20
FUTURE_FRAMES = 10

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

print("\nLoading LSTM model...")

model = load_model(MODEL_PATH)

print("LSTM model loaded!")

# ------------------------------------------------
# LOAD SCALER
# ------------------------------------------------

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded!")

# ------------------------------------------------
# LOAD CAMERA TRAJECTORY DATA
# ------------------------------------------------

CSV_PATH = "../data/processed/calibrated_camera_trajectories.csv"

df = pd.read_csv(CSV_PATH)

print("\nCamera trajectory data loaded!")

print("Rows:", len(df))
print("Columns:", list(df.columns))

# ------------------------------------------------
# CREATE ANGLE
# ------------------------------------------------

df["angle"] = np.arctan2(
    df["vy"],
    df["vx"]
)

# ------------------------------------------------
# PROCESS EACH OBJECT
# ------------------------------------------------

predictions = []

for track_id, object_data in df.groupby("track_id"):

    object_data = object_data.sort_values(
        "frame_id"
    )

    if len(object_data) < PAST_FRAMES:
        continue

    # Last 20 observations
    recent = object_data.tail(
        PAST_FRAMES
    ).copy()

    features = recent[
        ["world_x", "world_y", "vx", "vy", "angle"]
    ].values

    # ------------------------------------------------
    # SCALE CAMERA FEATURES
    # ------------------------------------------------

    scaled_features = scaler.transform(
        features
    )

    model_input = scaled_features[
        np.newaxis, :, :
    ].astype(np.float32)

    # ------------------------------------------------
    # LSTM PREDICTION
    # ------------------------------------------------

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

    # ------------------------------------------------
    # SAVE
    # ------------------------------------------------

    for step, point in enumerate(
        prediction,
        start=1
    ):

        predictions.append([
            track_id,
            object_data["object_type"].iloc[-1],
            step,
            point[0],
            point[1]
        ])


# ------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------

if len(predictions) == 0:

    print("\nEnough trajectory frames nahi mile.")
    print("At least 20 frames per object required.")

else:

    prediction_df = pd.DataFrame(
        predictions,
        columns=[
            "track_id",
            "object_type",
            "future_step",
            "predicted_x",
            "predicted_y"
        ]
    )

    output_path = (
        "../data/processed/"
        "camera_lstm_predictions.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False
    )

    print("\n========================================")
    print(" CAMERA LSTM PREDICTION COMPLETED")
    print("========================================")

    print("\nObjects predicted:",
          prediction_df["track_id"].nunique())

    print("Prediction rows:",
          len(prediction_df))

    print("\nSaved at:")
    print(output_path)

    print("\nFirst predictions:")
    print(
        prediction_df.head(20).to_string(
            index=False
        )
    )