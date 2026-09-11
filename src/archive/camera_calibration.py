import cv2
import numpy as np

print("========================================")
print(" CAMERA CALIBRATION")
print("========================================")

# Pixel coordinates from the uploaded camera image
pixel_points = np.float32([
    [560, 570],    # P1
    [1450, 570],   # P2
    [1530, 690],   # P3
    [480, 690]     # P4
])

# Approximate ground-plane coordinates
# Demo dimensions only
world_points = np.float32([
    [0, 0],        # P1
    [20, 0],       # P2
    [20, 8],       # P3
    [0, 8]         # P4
])

# Calculate perspective transformation
H, status = cv2.findHomography(
    pixel_points,
    world_points
)

print("\nHomography Matrix:")
print(H)


# Test point
test_pixel = np.float32([
    [[1000, 630]]
])

world_point = cv2.perspectiveTransform(
    test_pixel,
    H
)

print("\nTest Pixel Point:")
print("(1000, 630)")

print("\nConverted World Coordinate:")
print(world_point[0][0])

print("\n========================================")
print(" CALIBRATION COMPLETED")
print("========================================")