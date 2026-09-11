import os
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "camera_trajectory_sequences.npz"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "camera_lstm_trajectory.keras"
)

TARGET_SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "camera_target_scaler.pkl"
)

METRIC_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "results",
    "graphs"
)

os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading camera test data...")

data = np.load(DATA_FILE)

X_test = data["X_test"]
Y_test = data["Y_test"]

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading camera LSTM...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# LOAD TARGET SCALER
# ============================================================

target_scaler = joblib.load(
    TARGET_SCALER_FILE
)

target_min = target_scaler["min"]
target_max = target_scaler["max"]

target_range = target_max - target_min

target_range[target_range == 0] = 1.0


# ============================================================
# TARGET SCALING
# ============================================================

def scale_targets(y):

    return (
        y - target_min
    ) / target_range


def inverse_targets(y):

    return (
        y * target_range
    ) + target_min


# ============================================================
# SCALE TEST TARGET
# ============================================================

Y_test_scaled = scale_targets(
    Y_test
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

Y_pred_scaled = model.predict(
    X_test,
    verbose=1
)

# Convert back to world coordinates

Y_pred = inverse_targets(
    Y_pred_scaled
)


# ============================================================
# METRICS
# ============================================================

# Displacement error for every future timestep

errors = np.linalg.norm(
    Y_pred - Y_test,
    axis=2
)

# Average Displacement Error
ADE = np.mean(errors)

# Final Displacement Error
FDE = np.mean(
    errors[:, -1]
)

# RMSE
RMSE = np.sqrt(
    np.mean(
        (Y_pred - Y_test) ** 2
    )
)

# MAE
MAE = np.mean(
    np.abs(
        Y_pred - Y_test
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("CAMERA LSTM EVALUATION")
print("======================================")

print(f"ADE  : {ADE:.6f}")
print(f"FDE  : {FDE:.6f}")
print(f"RMSE : {RMSE:.6f}")
print(f"MAE  : {MAE:.6f}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Model": ["Camera LSTM"],
    "ADE": [ADE],
    "FDE": [FDE],
    "RMSE": [RMSE],
    "MAE": [MAE]
})

METRIC_FILE = os.path.join(
    METRIC_DIR,
    "camera_lstm_metrics.csv"
)

metrics.to_csv(
    METRIC_FILE,
    index=False
)

print("\nMetrics saved:")
print(METRIC_FILE)


# ============================================================
# TRAJECTORY VISUALIZATION
# ============================================================

sample_index = 0

actual = Y_test[sample_index]

predicted = Y_pred[sample_index]


plt.figure(figsize=(8, 6))

plt.plot(
    actual[:, 0],
    actual[:, 1],
    marker="o",
    label="Actual"
)

plt.plot(
    predicted[:, 0],
    predicted[:, 1],
    marker="x",
    label="Camera LSTM"
)

plt.xlabel("World X")

plt.ylabel("World Y")

plt.title(
    "Camera LSTM - Actual vs Predicted Trajectory"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


GRAPH_FILE = os.path.join(
    GRAPH_DIR,
    "camera_lstm_actual_vs_predicted.png"
)

plt.savefig(
    GRAPH_FILE,
    dpi=200
)

plt.close()


print("\nTrajectory graph saved:")
print(GRAPH_FILE)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    METRIC_DIR,
    "camera_lstm_predictions_test.npz"
)

np.savez_compressed(
    prediction_file,
    actual=Y_test,
    predicted=Y_pred
)

print("\nPredictions saved:")
print(prediction_file)


print("\n======================================")
print("CAMERA LSTM EVALUATION COMPLETE")
print("======================================")