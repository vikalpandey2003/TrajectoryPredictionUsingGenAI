import numpy as np
import matplotlib.pyplot as plt
import os


print("========================================")
print(" PROJECTILE TRAJECTORY PREDICTION")
print("========================================")


# ==========================================
# INPUT
# ==========================================

# Starting position
x0 = 0.0
y0 = 1.5

# Initial velocity
# vx = horizontal velocity
# vy = upward velocity
vx = 8.0
vy = 10.0

# Gravity
g = 9.81

# Time interval
dt = 0.05


# ==========================================
# CALCULATE FLIGHT TIME
# ==========================================

# y(t) = y0 + vy*t - 0.5*g*t^2

# Ground = y = 0

coefficients = [
    -0.5 * g,
    vy,
    y0
]

roots = np.roots(coefficients)

positive_roots = [
    r.real for r in roots
    if abs(r.imag) < 1e-8 and r.real > 0
]

if len(positive_roots) == 0:

    print("\nObject does not reach ground.")

    exit()

flight_time = max(positive_roots)


# ==========================================
# CREATE TIME VALUES
# ==========================================

times = np.arange(
    0,
    flight_time + dt,
    dt
)


# ==========================================
# PROJECTILE EQUATIONS
# ==========================================

x_positions = (
    x0 + vx * times
)

y_positions = (
    y0
    + vy * times
    - 0.5 * g * times ** 2
)


# Remove values below ground
y_positions = np.maximum(
    y_positions,
    0
)


# ==========================================
# LANDING POINT
# ==========================================

landing_x = (
    x0 + vx * flight_time
)

landing_y = 0


# ==========================================
# MAXIMUM HEIGHT
# ==========================================

time_to_max_height = vy / g

max_height = (
    y0
    + vy * time_to_max_height
    - 0.5 * g * time_to_max_height ** 2
)


# ==========================================
# RESULTS
# ==========================================

print("\n========================================")
print(" PROJECTILE RESULTS")
print("========================================")

print(
    "\nFlight Time:",
    round(flight_time, 3),
    "seconds"
)

print(
    "Landing X:",
    round(landing_x, 3),
    "meters"
)

print(
    "Landing Y:",
    landing_y,
    "meters"
)

print(
    "Maximum Height:",
    round(max_height, 3),
    "meters"
)

print(
    "Time to Maximum Height:",
    round(time_to_max_height, 3),
    "seconds"
)


# ==========================================
# SAVE GRAPH
# ==========================================

os.makedirs(
    "../results/graphs",
    exist_ok=True
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    x_positions,
    y_positions,
    linewidth=2,
    label="Predicted Projectile Path"
)

plt.scatter(
    [x0],
    [y0],
    s=80,
    label="Start"
)

plt.scatter(
    [landing_x],
    [landing_y],
    s=100,
    label="Predicted Landing Point"
)

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Height (m)")

plt.title(
    "Projectile Trajectory Prediction"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

output_path = (
    "../results/graphs/"
    "projectile_trajectory.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n========================================")
print(" PROJECTILE PREDICTION COMPLETED")
print("========================================")

print("\nGraph saved at:")
print(output_path)