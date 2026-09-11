import cv2
import csv
import os
import math
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR = "../data/processed"
CSV_PATH = "../data/processed/camera_trajectories.csv"

# Classes we want to track
TARGET_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "sports ball"
}

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("========================================")
print(" CAMERA OBJECT TRACKING")
print("========================================")

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")

# ==========================================
# OPEN CAMERA
# ==========================================

print("\nOpening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera opened successfully!")

# ==========================================
# CSV FILE
# ==========================================

csv_file = open(
    CSV_PATH,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "track_id",
    "frame_id",
    "timestamp_ms",
    "object_type",
    "x",
    "y",
    "vx",
    "vy",
    "speed"
])

# ==========================================
# TRAJECTORY MEMORY
# ==========================================

previous_positions = {}

frame_id = 0

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera frame could not be read.")
        break

    frame_id += 1

    timestamp_ms = int(
        cap.get(cv2.CAP_PROP_POS_MSEC)
    )

    # ======================================
    # OBJECT DETECTION + TRACKING
    # ======================================

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

                # Center of bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # ==================================
                # VELOCITY ESTIMATION
                # ==================================

                vx = 0.0
                vy = 0.0

                if track_id in previous_positions:

                    previous_x, previous_y, previous_time = \
                        previous_positions[track_id]

                    dt = (
                        timestamp_ms - previous_time
                    ) / 1000.0

                    if dt > 0:

                        vx = (
                            center_x - previous_x
                        ) / dt

                        vy = (
                            center_y - previous_y
                        ) / dt

                speed = math.sqrt(
                    vx ** 2 + vy ** 2
                )

                previous_positions[track_id] = (
                    center_x,
                    center_y,
                    timestamp_ms
                )

                # ==================================
                # SAVE TRAJECTORY DATA
                # ==================================

                csv_writer.writerow([
                    track_id,
                    frame_id,
                    timestamp_ms,
                    object_name,
                    center_x,
                    center_y,
                    vx,
                    vy,
                    speed
                ])

                # ==================================
                # DRAW TRACKING RESULT
                # ==================================

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

                cv2.circle(
                    frame,
                    (int(center_x), int(center_y)),
                    4,
                    (0, 0, 255),
                    -1
                )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        "Trajectory Tracking - Press Q to Exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Trajectory Prediction - Camera Tracking",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

csv_file.close()
cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print(" CAMERA TRACKING COMPLETED")
print("========================================")

print("\nTrajectory data saved at:")
print(CSV_PATH)