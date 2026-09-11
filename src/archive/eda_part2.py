import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# LOAD DATASET
# ==========================================

file_path = "../data/raw/vehicle_tracks_000.csv"

df = pd.read_csv(file_path)

print("======================================")
print(" TRAJECTORY EDA - PART 2")
print("======================================")

# ==========================================
# SPEED CALCULATION
# ==========================================

df["speed"] = np.sqrt(
    df["vx"] ** 2 + df["vy"] ** 2
)

print("\nSpeed Statistics:")
print(df["speed"].describe())

# ==========================================
# VEHICLE COUNT
# ==========================================

vehicle_counts = df["track_id"].value_counts()

print("\nTotal Vehicles:", df["track_id"].nunique())

print("\nTop 10 Vehicles by Number of Frames:")
print(vehicle_counts.head(10))

# ==========================================
# RESULTS FOLDER
# ==========================================

os.makedirs("../results/graphs", exist_ok=True)

# ==========================================
# GRAPH 1: MULTIPLE TRAJECTORIES
# ==========================================

top_vehicles = vehicle_counts.head(10).index

plt.figure(figsize=(10, 7))

for vehicle_id in top_vehicles:

    vehicle = df[df["track_id"] == vehicle_id].sort_values("frame_id")

    plt.plot(
        vehicle["x"],
        vehicle["y"],
        linewidth=1,
        label=f"Vehicle {vehicle_id}"
    )

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title("Multiple Vehicle Trajectories")

plt.legend()

plt.grid(True)

plt.savefig(
    "../results/graphs/multiple_vehicle_trajectories.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# GRAPH 2: SPEED DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["speed"],
    bins=50
)

plt.xlabel("Speed")
plt.ylabel("Frequency")

plt.title("Vehicle Speed Distribution")

plt.grid(True)

plt.savefig(
    "../results/graphs/speed_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n======================================")
print("EDA PART 2 COMPLETED")
print("======================================")

print("\nGraphs saved:")
print("1. multiple_vehicle_trajectories.png")
print("2. speed_distribution.png")