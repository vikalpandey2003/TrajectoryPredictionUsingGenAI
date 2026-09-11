import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================
# CONFIGURATION
# ============================================

DATA_PATH = "../data/processed/trajectory_sequences.npz"
MODEL_PATH = "../models/trajectory_generator.keras"

LATENT_DIM = 16
K = 20


print("========================================")
print(" FULL GAN TRAJECTORY EVALUATION")
print("========================================")


# ============================================
# LOAD TEST DATA
# ============================================

data = np.load(DATA_PATH)

X_test = data["X_test"]
Y_test = data["Y_test"]

print("\nTest Dataset:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ============================================
# LOAD GENERATOR
# ============================================

print("\nLoading GAN Generator...")

generator = load_model(MODEL_PATH)

print("Generator loaded successfully!")


# ============================================
# BEST-OF-K EVALUATION
# ============================================

num_samples = len(X_test)

print("\nTotal test sequences:", num_samples)
print("Generating", K, "trajectories per sample...")


all_ade = []
all_fde = []
all_rmse = []
all_mae = []


for i in range(num_samples):

    past = X_test[i:i+1]
    actual = Y_test[i:i+1]

    best_ade = float("inf")
    best_fde = float("inf")
    best_rmse = float("inf")
    best_mae = float("inf")

    for k in range(K):

        noise = np.random.normal(
            0,
            1,
            size=(1, LATENT_DIM)
        ).astype(np.float32)

        prediction = generator.predict(
            [past, noise],
            verbose=0
        )

        # ----------------------------------------
        # Displacement error
        # ----------------------------------------

        displacement = np.sqrt(
            (prediction[:, :, 0] - actual[:, :, 0]) ** 2
            +
            (prediction[:, :, 1] - actual[:, :, 1]) ** 2
        )

        ade = np.mean(displacement)

        fde = displacement[:, -1].mean()

        rmse = np.sqrt(
            mean_squared_error(
                actual.reshape(-1, 2),
                prediction.reshape(-1, 2)
            )
        )

        mae = mean_absolute_error(
            actual.reshape(-1, 2),
            prediction.reshape(-1, 2)
        )

        # ----------------------------------------
        # Select best trajectory
        # ----------------------------------------

        if ade < best_ade:

            best_ade = ade
            best_fde = fde
            best_rmse = rmse
            best_mae = mae


    all_ade.append(best_ade)
    all_fde.append(best_fde)
    all_rmse.append(best_rmse)
    all_mae.append(best_mae)


    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{num_samples}"
        )


# ============================================
# FINAL RESULTS
# ============================================

final_ade = np.mean(all_ade)
final_fde = np.mean(all_fde)
final_rmse = np.mean(all_rmse)
final_mae = np.mean(all_mae)


print("\n========================================")
print(" FINAL GAN RESULTS")
print("========================================")

print("\nBest-of-", K, "Evaluation")

print("\nADE  :", final_ade)
print("FDE  :", final_fde)
print("RMSE :", final_rmse)
print("MAE  :", final_mae)


# ============================================
# SAVE RESULTS
# ============================================

os.makedirs(
    "../results/metrics",
    exist_ok=True
)

results = pd.DataFrame({

    "Model": ["GAN"],

    "Evaluation": [
        "Best-of-20"
    ],

    "ADE": [
        final_ade
    ],

    "FDE": [
        final_fde
    ],

    "RMSE": [
        final_rmse
    ],

    "MAE": [
        final_mae
    ]
})


results.to_csv(
    "../results/metrics/gan_metrics.csv",
    index=False
)


print("\nMetrics saved at:")

print(
    "../results/metrics/gan_metrics.csv"
)


print("\n========================================")
print(" GAN EVALUATION COMPLETED")
print("========================================")