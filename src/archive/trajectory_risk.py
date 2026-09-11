import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import load_model


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "../data/processed/trajectory_sequences.npz"
MODEL_PATH = "../models/lstm_trajectory.keras"

FUTURE_FRAME_TIME = 0.1

# Target point
# Later camera system se dynamically milega
TARGET_X = 0.70
TARGET_Y = 0.70

TARGET_THRESHOLD = 0.05


print("========================================")
print(" TRAJECTORY RISK ANALYSIS")
print("========================================")


# ==========================================
# LOAD DATA
# ==========================================

data = np.load(DATA_PATH)

X_test = data["X_test"]
Y_test = data["Y_test"]

print("\nTest data loaded!")

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ==========================================
# LOAD LSTM
# ==========================================

print("\nLoading LSTM model...")

model = load_model(MODEL_PATH)

print("LSTM loaded successfully!")


# ==========================================
# SELECT OBJECT
# ==========================================

sample_index = 0

past = X_test[
    sample_index:sample_index + 1
]

actual = Y_test[
    sample_index
]

print("\nSelected trajectory:", sample_index)


# ==========================================
# PREDICT FUTURE
# ==========================================

prediction = model.predict(
    past,
    verbose=0
)[0]

print("\nFuture trajectory predicted!")

print(
    "Prediction shape:",
    prediction.shape
)


# ==========================================
# TIME TO TARGET
# ==========================================

print("\n========================================")
print(" TIME TO TARGET")
print("========================================")

arrival_time = None
arrival_position = None


for i, point in enumerate(prediction):

    x = point[0]
    y = point[1]

    distance = np.sqrt(
        (x - TARGET_X) ** 2 +
        (y - TARGET_Y) ** 2
    )

    if distance <= TARGET_THRESHOLD:

        arrival_time = (
            (i + 1) *
            FUTURE_FRAME_TIME
        )

        arrival_position = (
            x,
            y
        )

        break


if arrival_time is not None:

    print("\nTarget reached!")

    print(
        "Predicted X:",
        arrival_position[0]
    )

    print(
        "Predicted Y:",
        arrival_position[1]
    )

    print(
        "Estimated Time:",
        round(arrival_time, 2),
        "seconds"
    )

else:

    print("\nTarget was NOT reached")
    print("within prediction horizon.")


# ==========================================
# CLOSEST APPROACH
# ==========================================

print("\n========================================")
print(" CLOSEST APPROACH")
print("========================================")

distances = []

for point in prediction:

    distance = np.sqrt(
        (point[0] - TARGET_X) ** 2 +
        (point[1] - TARGET_Y) ** 2
    )

    distances.append(distance)


distances = np.array(distances)

closest_index = np.argmin(distances)

closest_distance = distances[
    closest_index
]

closest_time = (
    (closest_index + 1) *
    FUTURE_FRAME_TIME
)

closest_position = prediction[
    closest_index
]


print(
    "Closest distance:",
    closest_distance
)

print(
    "Closest X:",
    closest_position[0]
)

print(
    "Closest Y:",
    closest_position[1]
)

print(
    "Time of closest approach:",
    round(closest_time, 2),
    "seconds"
)


# ==========================================
# RISK LEVEL
# ==========================================

print("\n========================================")
print(" RISK ANALYSIS")
print("========================================")


if closest_distance <= TARGET_THRESHOLD:

    risk = "HIGH"

elif closest_distance <= TARGET_THRESHOLD * 3:

    risk = "MEDIUM"

else:

    risk = "LOW"


print(
    "Risk Level:",
    risk
)


# ==========================================
# SAVE RESULT
# ==========================================

os.makedirs(
    "../results/metrics",
    exist_ok=True
)

result = pd.DataFrame({

    "Sample": [sample_index],

    "Target_X": [TARGET_X],

    "Target_Y": [TARGET_Y],

    "Closest_Distance": [
        closest_distance
    ],

    "Closest_Approach_Time": [
        closest_time
    ],

    "Risk": [risk]

})


if arrival_time is not None:

    result["Time_To_Target"] = [
        arrival_time
    ]

else:

    result["Time_To_Target"] = [
        np.nan
    ]


result.to_csv(
    "../results/metrics/trajectory_risk.csv",
    index=False
)


print("\nResult saved at:")

print(
    "../results/metrics/trajectory_risk.csv"
)


print("\n========================================")
print(" RISK ANALYSIS COMPLETED")
print("========================================")