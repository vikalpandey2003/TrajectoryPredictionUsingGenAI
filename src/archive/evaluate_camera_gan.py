import os
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import csv

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
    "camera_trajectory_generator.keras"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "camera_gan_target_scaler.pkl"
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
# SETTINGS
# ============================================================

FUTURE_FRAMES = 10
LATENT_DIM = 16
NUM_GENERATIONS = 20


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
# LOAD GENERATOR
# ============================================================

print("\nLoading Camera GAN Generator...")

generator = tf.keras.models.load_model(
    MODEL_FILE
)

print("Generator loaded successfully.")


# ============================================================
# LOAD TARGET SCALER
# ============================================================

print("\nLoading target scaler...")

scaler_data = joblib.load(
    SCALER_FILE
)

target_min = scaler_data["min"]
target_max = scaler_data["max"]

target_range = target_max - target_min

target_range[target_range == 0] = 1.0

print("Target min:", target_min)
print("Target max:", target_max)


# ============================================================
# INVERSE SCALING
# ============================================================

def inverse_scale(y_scaled):

    return (
        y_scaled * target_range
    ) + target_min


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_ade(actual, predicted):

    distances = np.sqrt(
        np.sum(
            (actual - predicted) ** 2,
            axis=1
        )
    )

    return np.mean(distances)


def calculate_fde(actual, predicted):

    return np.linalg.norm(
        actual[-1] - predicted[-1]
    )


def calculate_rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


# ============================================================
# EVALUATION
# ============================================================

print("\n======================================")
print("CAMERA GAN EVALUATION STARTED")
print("======================================")

all_ade = []
all_fde = []
all_rmse = []
all_mae = []

best_predictions = []
best_actuals = []

for i in range(len(X_test)):

    past_sample = X_test[
        i:i + 1
    ]

    actual_scaled = Y_test[i]

    # Convert actual target from scaled/world representation
    actual_world = actual_scaled

    best_ade = float("inf")
    best_prediction = None
    best_fde = None
    best_rmse = None
    best_mae = None

    # --------------------------------------------------------
    # Generate 20 candidate trajectories
    # --------------------------------------------------------

    for _ in range(NUM_GENERATIONS):

        noise = tf.random.normal(
            [1, LATENT_DIM]
        )

        past_tensor = tf.convert_to_tensor(
            past_sample,
            dtype=tf.float32
        )

        generated_scaled = generator(
            [past_tensor, noise],
            training=False
        ).numpy()[0]

        generated_world = inverse_scale(
            generated_scaled
        )

        # ----------------------------------------------------
        # Calculate ADE
        # ----------------------------------------------------

        ade = calculate_ade(
            actual_world,
            generated_world
        )

        if ade < best_ade:

            best_ade = ade
            best_prediction = generated_world

            best_fde = calculate_fde(
                actual_world,
                generated_world
            )

            best_rmse = calculate_rmse(
                actual_world,
                generated_world
            )

            best_mae = calculate_mae(
                actual_world,
                generated_world
            )

    all_ade.append(best_ade)
    all_fde.append(best_fde)
    all_rmse.append(best_rmse)
    all_mae.append(best_mae)

    best_predictions.append(
        best_prediction
    )

    best_actuals.append(
        actual_world
    )

    if (i + 1) % 50 == 0:

        print(
            f"Processed {i + 1}/{len(X_test)} test samples"
        )


# ============================================================
# FINAL METRICS
# ============================================================

final_ade = np.mean(all_ade)
final_fde = np.mean(all_fde)
final_rmse = np.mean(all_rmse)
final_mae = np.mean(all_mae)


print("\n======================================")
print("CAMERA GAN RESULTS")
print("======================================")

print(
    f"ADE  : {final_ade:.6f}"
)

print(
    f"FDE  : {final_fde:.6f}"
)

print(
    f"RMSE : {final_rmse:.6f}"
)

print(
    f"MAE  : {final_mae:.6f}"
)


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_file = os.path.join(
    METRIC_DIR,
    "camera_gan_metrics.csv"
)

with open(
    metrics_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Evaluation",
        "ADE",
        "FDE",
        "RMSE",
        "MAE"
    ])

    writer.writerow([
        "Camera GAN",
        "Best-of-20",
        final_ade,
        final_fde,
        final_rmse,
        final_mae
    ])


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    METRIC_DIR,
    "camera_gan_predictions_test.npz"
)

np.savez(
    prediction_file,
    predictions=np.array(
        best_predictions
    ),
    actual=np.array(
        best_actuals
    )
)


# ============================================================
# SAMPLE GRAPH
# ============================================================

sample_index = 0

actual = best_actuals[
    sample_index
]

prediction = best_predictions[
    sample_index
]

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    actual[:, 0],
    actual[:, 1],
    marker="o",
    label="Actual"
)

plt.plot(
    prediction[:, 0],
    prediction[:, 1],
    marker="x",
    label="Camera GAN"
)

plt.xlabel("World X")
plt.ylabel("World Y")

plt.title(
    "Camera GAN - Actual vs Predicted Trajectory"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

graph_file = os.path.join(
    GRAPH_DIR,
    "camera_gan_actual_vs_predicted.png"
)

plt.savefig(
    graph_file,
    dpi=200
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nMetrics saved:")
print(metrics_file)

print("\nPredictions saved:")
print(prediction_file)

print("\nGraph saved:")
print(graph_file)

print("\n======================================")
print("CAMERA GAN EVALUATION COMPLETE")
print("======================================")