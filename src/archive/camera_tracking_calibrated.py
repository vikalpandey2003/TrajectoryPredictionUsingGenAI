import cv2
import csv
import os
import math
import numpy as np
from ultralytics import YOLO

print("========================================")
print(" CALIBRATED CAMERA TRAJECTORY TRACKING")
print("========================================")

MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "../data/processed/calibrated_camera_trajectories.csv"

os.makedirs("../data/processed", exist_ok=True)

# ------------------------------------------------
# CAMERA CALIBRATION
# ------------------------------------------------

pixel_points = np.float32([
    [560, 570],     # P1
    [1450, 570],    # P2
    [1530, 690],    # P3
    [480, 690]      # P4
])

# Approximate ground-plane coordinates
# These are demo dimensions, not measured real-world values.
world_points = np.float32([
    [0, 0],
    [20, 0],
    [20, 8],
    [0, 8]
])

H, status = cv2.findHomography(
    pixel_points,
    world_points
)

print("\nHomography matrix calculated.")

# ------------------------------------------------
# YOLO MODEL
# ------------------------------------------------

model = YOLO(MODEL_PATH)

TARGET_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "sports ball"
}

# ------------------------------------------------
# CAMERA
# ------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi hua.")
    exit()

# ------------------------------------------------
# CSV
# ------------------------------------------------

csv_file = open(
    OUTPUT_PATH,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "track_id",
    "frame_id",
    "timestamp_ms",
    "object_type",
    "pixel_x",
    "pixel_y",
    "world_x",
    "world_y",
    "vx",
    "vy",
    "speed"
])

# Previous world positions
previous_positions = {}

frame_id = 0

print("\nCamera started.")
print("Tracking started...")
print("Press Q to exit.\n")

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Frame read nahi hua.")
        break

    frame_id += 1

    timestamp_ms = int(
        cap.get(cv2.CAP_PROP_POS_MSEC)
    )

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()

            for track_id, class_id, box in zip(
                track_ids,
                class_ids,
                coordinates
            ):

                object_name = model.names[class_id]

                if object_name not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                # Bottom-center of bounding box
                # Better approximation of object's ground contact point
                pixel_x = (x1 + x2) / 2
                pixel_y = y2

                # ------------------------------------------------
                # PIXEL -> WORLD COORDINATE
                # ------------------------------------------------

                pixel_point = np.float32([
                    [[pixel_x, pixel_y]]
                ])

                world_point = cv2.perspectiveTransform(
                    pixel_point,
                    H
                )

                world_x = float(
                    world_point[0][0][0]
                )

                world_y = float(
                    world_point[0][0][1]
                )

                # ------------------------------------------------
                # WORLD VELOCITY
                # ------------------------------------------------

                vx = 0.0
                vy = 0.0

                if track_id in previous_positions:

                    old_x, old_y, old_time = \
                        previous_positions[track_id]

                    dt = (
                        timestamp_ms - old_time
                    ) / 1000.0

                    if dt > 0:

                        vx = (
                            world_x - old_x
                        ) / dt

                        vy = (
                            world_y - old_y
                        ) / dt

                speed = math.sqrt(
                    vx ** 2 + vy ** 2
                )

                previous_positions[track_id] = (
                    world_x,
                    world_y,
                    timestamp_ms
                )

                # ------------------------------------------------
                # SAVE DATA
                # ------------------------------------------------

                writer.writerow([
                    track_id,
                    frame_id,
                    timestamp_ms,
                    object_name,
                    pixel_x,
                    pixel_y,
                    world_x,
                    world_y,
                    vx,
                    vy,
                    speed
                ])

                # ------------------------------------------------
                # DISPLAY
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{object_name} "
                    f"ID:{track_id}"
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                coordinate_text = (
                    f"X:{world_x:.2f} "
                    f"Y:{world_y:.2f}"
                )

                cv2.putText(
                    frame,
                    coordinate_text,
                    (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.circle(
                    frame,
                    (int(pixel_x), int(pixel_y)),
                    5,
                    (0, 0, 255),
                    -1
                )

    cv2.putText(
        frame,
        "Calibrated Trajectory Tracking - Press Q",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Calibrated Camera Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------------
# CLEANUP
# ------------------------------------------------

csv_file.close()
cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print(" TRACKING COMPLETED")
print("========================================")

print("\nSaved file:")
print(OUTPUT_PATH)