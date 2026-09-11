import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("========================================")
print(" LSTM FULL TEST EVALUATION")
print("========================================")

DATA_PATH = "../data/processed/trajectory_sequences.npz"
MODEL_PATH = "../models/lstm_trajectory.keras"

data = np.load(DATA_PATH)

X_test = data["X_test"]
Y_test = data["Y_test"]

print("\nTest data loaded!")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

print("\nLoading trained LSTM model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nGenerating predictions for complete test set...")

predictions = model.predict(
    X_test,
    batch_size=64,
    verbose=1
)

print("\nPrediction shape:")
print(predictions.shape)

# -----------------------------------
# Calculate displacement error
# -----------------------------------

errors = np.sqrt(
    (predictions[:, :, 0] - Y_test[:, :, 0]) ** 2 +
    (predictions[:, :, 1] - Y_test[:, :, 1]) ** 2
)

# ADE
ade = np.mean(errors)

# FDE
fde = np.mean(errors[:, -1])

# RMSE
rmse = np.sqrt(
    mean_squared_error(
        Y_test.reshape(-1, 2),
        predictions.reshape(-1, 2)
    )
)

# MAE
mae = mean_absolute_error(
    Y_test.reshape(-1, 2),
    predictions.reshape(-1, 2)
)

print("\n========================================")
print(" FINAL LSTM EVALUATION")
print("========================================")

print("\nNumber of test sequences:", len(X_test))

print("\nADE  :", ade)
print("FDE  :", fde)
print("RMSE :", rmse)
print("MAE  :", mae)

# -----------------------------------
# Save metrics
# -----------------------------------

os.makedirs("../results/metrics", exist_ok=True)

metrics = pd.DataFrame({
    "Model": ["LSTM"],
    "ADE": [ade],
    "FDE": [fde],
    "RMSE": [rmse],
    "MAE": [mae]
})

metrics.to_csv(
    "../results/metrics/lstm_metrics.csv",
    index=False
)

print("\nMetrics saved at:")
print("../results/metrics/lstm_metrics.csv")

print("\n========================================")
print(" LSTM EVALUATION COMPLETED")
print("========================================")