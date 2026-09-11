import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# SETTINGS
# ==========================================

DATA_PATH = "../data/processed/trajectory_sequences.npz"

PAST_FRAMES = 20
FUTURE_FRAMES = 10
FEATURES = 5


# ==========================================
# LOAD DATA
# ==========================================

print("========================================")
print(" LSTM TRAJECTORY PREDICTION")
print("========================================")

data = np.load(DATA_PATH)

X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

X_test = data["X_test"]
Y_test = data["Y_test"]


print("\nDataset loaded!")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ==========================================
# BUILD LSTM ENCODER-DECODER
# ==========================================

model = Sequential()

# Encoder
model.add(
    LSTM(
        128,
        activation="tanh",
        input_shape=(PAST_FRAMES, FEATURES)
    )
)

# Convert encoder representation
# into future sequence length
model.add(
    RepeatVector(FUTURE_FRAMES)
)

# Decoder
model.add(
    LSTM(
        128,
        activation="tanh",
        return_sequences=True
    )
)

# Output X,Y
model.add(
    TimeDistributed(
        Dense(2)
    )
)


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="mse"
)


print("\n========================================")
print(" MODEL SUMMARY")
print("========================================")

model.summary()


# ==========================================
# CREATE MODEL FOLDER
# ==========================================

os.makedirs(
    "../models",
    exist_ok=True
)

os.makedirs(
    "../results/graphs",
    exist_ok=True
)


# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "../models/lstm_trajectory.keras",
    monitor="val_loss",
    save_best_only=True
)


# ==========================================
# TRAIN
# ==========================================

print("\n========================================")
print(" STARTING LSTM TRAINING")
print("========================================")

history = model.fit(
    X_train,
    Y_train,

    validation_data=(
        X_val,
        Y_val
    ),

    epochs=30,

    batch_size=64,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# ==========================================
# TEST EVALUATION
# ==========================================

print("\n========================================")
print(" TEST EVALUATION")
print("========================================")

test_loss = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)

print("Test MSE:", test_loss)


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

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

plt.title("LSTM Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "../results/graphs/lstm_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# PREDICTION
# ==========================================

print("\nGenerating prediction...")

predictions = model.predict(
    X_test[:1],
    verbose=0
)

actual = Y_test[0]


# ==========================================
# TRAJECTORY GRAPH
# ==========================================

plt.figure(figsize=(10, 7))

# Historical trajectory
plt.plot(
    X_test[0, :, 0],
    X_test[0, :, 1],
    marker="o",
    label="Historical"
)

# Actual future
plt.plot(
    actual[:, 0],
    actual[:, 1],
    marker="o",
    label="Actual Future"
)

# Predicted future
plt.plot(
    predictions[0, :, 0],
    predictions[0, :, 1],
    marker="x",
    linestyle="--",
    label="LSTM Prediction"
)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    "LSTM Vehicle Trajectory Prediction"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "../results/graphs/lstm_prediction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# ADE
# ==========================================

displacement = np.sqrt(
    (predictions[0, :, 0] - actual[:, 0]) ** 2
    +
    (predictions[0, :, 1] - actual[:, 1]) ** 2
)

ade = np.mean(displacement)


# ==========================================
# FDE
# ==========================================

fde = displacement[-1]


print("\n========================================")
print(" LSTM RESULTS")
print("========================================")

print("Test MSE :", test_loss)
print("ADE      :", ade)
print("FDE      :", fde)

print("\nModel saved at:")
print("../models/lstm_trajectory.keras")

print("\nGraphs saved:")
print("../results/graphs/lstm_training_loss.png")
print("../results/graphs/lstm_prediction.png")