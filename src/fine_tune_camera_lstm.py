import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

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

OLD_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "lstm_trajectory.keras"
)

NEW_MODEL = os.path.join(
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

GRAPH_FILE = os.path.join(
    BASE_DIR,
    "results",
    "graphs",
    "camera_lstm_fine_tuning.png"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading camera dataset...")

data = np.load(DATA_FILE)

X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

X_test = data["X_test"]
Y_test = data["Y_test"]

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

# ============================================================
# TARGET NORMALIZATION
# ============================================================

print("\nNormalizing target trajectories...")

target_min = Y_train.reshape(-1, 2).min(axis=0)
target_max = Y_train.reshape(-1, 2).max(axis=0)

target_range = target_max - target_min

target_range[target_range == 0] = 1.0


def scale_targets(y):
    return (
        y - target_min
    ) / target_range


Y_train_scaled = scale_targets(Y_train)
Y_val_scaled = scale_targets(Y_val)
Y_test_scaled = scale_targets(Y_test)

# Save target scaling information
target_scaler = {
    "min": target_min,
    "max": target_max
}

joblib.dump(
    target_scaler,
    TARGET_SCALER_FILE
)

print("Target scaler saved.")

# ============================================================
# LOAD EXISTING LSTM
# ============================================================

print("\nLoading original LSTM model...")

model = tf.keras.models.load_model(
    OLD_MODEL,
    compile=False
)

print("Original LSTM loaded successfully.")

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00005
    ),
    loss="mse",
    metrics=["mae"]
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    NEW_MODEL,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\n======================================")
print("CAMERA LSTM FINE-TUNING STARTED")
print("======================================")

history = model.fit(
    X_train,
    Y_train_scaled,
    validation_data=(
        X_val,
        Y_val_scaled
    ),
    epochs=30,
    batch_size=32,
    callbacks=[
        checkpoint,
        early_stop
    ],
    verbose=1
)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n======================================")
print("TEST EVALUATION")
print("======================================")

test_loss, test_mae = model.evaluate(
    X_test,
    Y_test_scaled,
    verbose=1
)

print("\nCamera LSTM Test Loss:", test_loss)
print("Camera LSTM Test MAE:", test_mae)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(NEW_MODEL)

print("\nFine-tuned model saved:")
print(NEW_MODEL)

# ============================================================
# TRAINING GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Camera LSTM Fine-Tuning"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=200
)

plt.close()

print("\nTraining graph saved:")
print(GRAPH_FILE)

print("\n======================================")
print("CAMERA LSTM FINE-TUNING COMPLETE")
print("======================================")