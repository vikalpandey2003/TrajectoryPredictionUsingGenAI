import numpy as np

DATA_PATH = "../data/processed/trajectory_sequences.npz"

print("========================================")
print(" LSTM DATA SCALING CHECK")
print("========================================")

data = np.load(DATA_PATH)

X_train = data["X_train"]
Y_train = data["Y_train"]

X_test = data["X_test"]
Y_test = data["Y_test"]

print("\nX_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("\nX_train range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())

print("\nY_train range:")
print("Minimum:", Y_train.min())
print("Maximum:", Y_train.max())

print("\nX_test range:")
print("Minimum:", X_test.min())
print("Maximum:", X_test.max())

print("\nY_test range:")
print("Minimum:", Y_test.min())
print("Maximum:", Y_test.max())

print("\n========================================")

if X_train.min() >= 0 and X_train.max() <= 1:
    print("X data appears NORMALIZED.")
else:
    print("X data appears to contain RAW values.")

if Y_train.min() >= 0 and Y_train.max() <= 1:
    print("Y data appears NORMALIZED.")
else:
    print("Y data appears to contain RAW values.")

print("========================================")