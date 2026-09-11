import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "trajectory_sequences.npz"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "scaler.pkl"
)

LSTM_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm_trajectory.keras"
)

GAN_PATH = os.path.join(
    BASE_DIR,
    "models",
    "trajectory_generator.keras"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "graphs",
    "final_clean_trajectory_comparison.png"
)


# ============================================================
# SETTINGS
# ============================================================

PAST_FRAMES = 20
FUTURE_FRAMES = 10
NOISE_DIM = 16
GAN_SAMPLES = 20

# Select ONE test sample
SAMPLE_INDEX = 0


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading test data...")

data = np.load(
    DATA_PATH
)

X_test = data["X_test"]

Y_test = data["Y_test"]


print(
    "X_test shape:",
    X_test.shape
)

print(
    "Y_test shape:",
    Y_test.shape
)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading LSTM...")

lstm_model = tf.keras.models.load_model(
    LSTM_PATH,
    compile=False
)


print("Loading GAN...")

gan_model = tf.keras.models.load_model(
    GAN_PATH,
    compile=False
)


# ============================================================
# SELECT SAMPLE
# ============================================================

past_scaled = X_test[
    SAMPLE_INDEX
]

actual_scaled = Y_test[
    SAMPLE_INDEX
]


# ============================================================
# LSTM PREDICTION
# ============================================================

print("\nGenerating LSTM prediction...")

lstm_scaled = lstm_model.predict(
    past_scaled[np.newaxis, ...],
    verbose=0
)[0]


# ============================================================
# GAN PREDICTIONS
# ============================================================

print("Generating GAN trajectories...")

past_batch = np.repeat(
    past_scaled[np.newaxis, ...],
    GAN_SAMPLES,
    axis=0
)


noise = np.random.normal(
    0,
    1,
    size=(
        GAN_SAMPLES,
        NOISE_DIM
    )
).astype(
    np.float32
)


gan_scaled = gan_model.predict(
    [
        past_batch,
        noise
    ],
    verbose=0
)


# ============================================================
# INVERSE SCALING
# ============================================================

def inverse_xy(
    values
):

    values = np.asarray(
        values
    )

    result = np.zeros_like(
        values,
        dtype=np.float32
    )

    # x
    result[..., 0] = (
        values[..., 0]
        * (
            scaler.data_max_[0]
            - scaler.data_min_[0]
        )
        + scaler.data_min_[0]
    )

    # y
    result[..., 1] = (
        values[..., 1]
        * (
            scaler.data_max_[1]
            - scaler.data_min_[1]
        )
        + scaler.data_min_[1]
    )

    return result


# ============================================================
# CONVERT TO ORIGINAL COORDINATES
# ============================================================

past = inverse_xy(
    past_scaled[:, :2]
)

actual = inverse_xy(
    actual_scaled
)

lstm_prediction = inverse_xy(
    lstm_scaled
)

gan_predictions = inverse_xy(
    gan_scaled
)


# ============================================================
# FIND BEST GAN
# ============================================================

gan_ade = np.mean(
    np.sqrt(
        np.sum(
            (
                gan_predictions
                - actual[np.newaxis, :, :]
            ) ** 2,
            axis=2
        )
    ),
    axis=1
)


best_index = np.argmin(
    gan_ade
)


best_gan = gan_predictions[
    best_index
]


best_gan_ade = gan_ade[
    best_index
]


# ============================================================
# CALCULATE LSTM ERROR
# ============================================================

lstm_ade = np.mean(
    np.sqrt(
        np.sum(
            (
                lstm_prediction
                - actual
            ) ** 2,
            axis=1
        )
    )
)


lstm_fde = np.sqrt(
    np.sum(
        (
            lstm_prediction[-1]
            - actual[-1]
        ) ** 2
    )
)


gan_fde = np.sqrt(
    np.sum(
        (
            best_gan[-1]
            - actual[-1]
        ) ** 2
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("SINGLE SAMPLE VISUALIZATION")
print("========================================")

print(
    f"LSTM ADE : {lstm_ade:.6f}"
)

print(
    f"LSTM FDE : {lstm_fde:.6f}"
)

print(
    f"GAN ADE  : {best_gan_ade:.6f}"
)

print(
    f"GAN FDE  : {gan_fde:.6f}"
)

print(
    "Best GAN sample:",
    best_index + 1
)


# ============================================================
# CREATE CLEAN GRAPH
# ============================================================

plt.figure(
    figsize=(12, 8)
)


# ------------------------------------------------------------
# Historical
# ------------------------------------------------------------

plt.plot(
    past[:, 0],
    past[:, 1],
    marker="o",
    linewidth=2.5,
    label="Historical Trajectory"
)


# ------------------------------------------------------------
# Actual future
# ------------------------------------------------------------

plt.plot(
    actual[:, 0],
    actual[:, 1],
    marker="o",
    linewidth=3,
    label="Actual Future"
)


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

plt.plot(
    lstm_prediction[:, 0],
    lstm_prediction[:, 1],
    marker="x",
    linewidth=3,
    label="LSTM Prediction"
)


# ------------------------------------------------------------
# Best GAN
# ------------------------------------------------------------

plt.plot(
    best_gan[:, 0],
    best_gan[:, 1],
    marker="s",
    linewidth=3,
    linestyle="--",
    label="Best GAN Prediction"
)


# ------------------------------------------------------------
# Starting point
# ------------------------------------------------------------

plt.scatter(
    past[-1, 0],
    past[-1, 1],
    s=120,
    marker="o",
    label="Prediction Start"
)


# ============================================================
# LABELS
# ============================================================

plt.title(
    "Trajectory Prediction: LSTM vs GAN",
    fontsize=18
)

plt.xlabel(
    "X Position",
    fontsize=13
)

plt.ylabel(
    "Y Position",
    fontsize=13
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend(
    fontsize=11
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "\nClean graph saved successfully:"
)

print(
    OUTPUT_PATH
)