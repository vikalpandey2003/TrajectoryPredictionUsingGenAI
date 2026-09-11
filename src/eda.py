import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. LOAD DATASET
# ==============================

file_path = "../data/raw/vehicle_tracks_000.csv"

df = pd.read_csv(file_path)

print("================================")
print(" TRAJECTORY DATASET - EDA")
print("================================")

# ==============================
# 2. BASIC INFORMATION
# ==============================

print("\nTotal Rows:", len(df))
print("Total Columns:", len(df.columns))

print("\nUnique Vehicles:")
print(df["track_id"].nunique())

print("\nAgent Types:")
print(df["agent_type"].value_counts())

# ==============================
# 3. FRAMES PER VEHICLE
# ==============================

frames_per_vehicle = df.groupby("track_id")["frame_id"].count()

print("\nFrames per Vehicle:")
print(frames_per_vehicle.describe())

# ==============================
# 4. POSITION STATISTICS
# ==============================

print("\nX Position Statistics:")
print(df["x"].describe())

print("\nY Position Statistics:")
print(df["y"].describe())

# ==============================
# 5. VELOCITY STATISTICS
# ==============================

print("\nVelocity X Statistics:")
print(df["vx"].describe())

print("\nVelocity Y Statistics:")
print(df["vy"].describe())

# ==============================
# 6. CREATE RESULTS FOLDER
# ==============================

os.makedirs("../results/graphs", exist_ok=True)

# ==============================
# 7. SELECT ONE VEHICLE
# ==============================

vehicle_id = df["track_id"].value_counts().idxmax()

vehicle_data = df[df["track_id"] == vehicle_id].sort_values("frame_id")

print("\nVehicle selected for visualization:", vehicle_id)
print("Number of frames:", len(vehicle_data))

# ==============================
# 8. TRAJECTORY PLOT
# ==============================

plt.figure(figsize=(10, 6))

plt.plot(
    vehicle_data["x"],
    vehicle_data["y"],
    marker=".",
    markersize=3
)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    f"Vehicle Trajectory - Track ID {vehicle_id}"
)

plt.grid(True)

# ==============================
# 9. SAVE GRAPH
# ==============================

output_path = "../results/graphs/vehicle_trajectory.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("\nTrajectory graph saved at:")
print(output_path)

plt.show() 