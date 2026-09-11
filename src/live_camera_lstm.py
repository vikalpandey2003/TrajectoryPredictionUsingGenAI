import cv2
import numpy as np
import math
import joblib
from collections import defaultdict, deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model

print("========================================")
print(" LIVE CAMERA + LSTM TRAJECTORY")
print("========================================")

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

YOLO_MODEL = "yolo11n.pt"
LSTM_MODEL = "../models/lstm_trajectory.keras"
SCALER_PATH = "../data/processed/scaler.pkl"

PAST_FRAMES = 20
FUTURE_FRAMES = 10

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

print("\nLoading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading LSTM...")
lstm = load_model(LSTM_MODEL)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("All models loaded successfully!")

# ------------------------------------------------
# CAMERA CALIBRATION
# ------------------------------------------------

pixel_points = np.float32([
    [560, 570],
    [1450, 570],
    [1530, 690],
    [480, 690]
])

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

# ------------------------------------------------
# OBJECT CLASSES
# ------------------------------------------------

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
# TRAJECTORY HISTORY
# ------------------------------------------------

history = defaultdict(
    lambda: deque(maxlen=PAST_FRAMES)
)

previous_data = {}

# ------------------------------------------------
# CAMERA
# ------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nCamera open nahi hua.")
    exit()

print("\n========================================")
print(" CAMERA STARTED")
print(" Move an object in front of camera")
print(" Press Q to exit")
print("========================================")

# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    timestamp = cv2.getTickCount() / cv2.getTickFrequency()

    results = yolo.track(
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

                object_name = yolo.names[class_id]

                if object_name not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                # Ground contact point
                pixel_x = (x1 + x2) / 2
                pixel_y = y2

                # ------------------------------------------------
                # PIXEL → WORLD
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
                # VELOCITY
                # ------------------------------------------------

                vx = 0.0
                vy = 0.0

                if track_id in previous_data:

                    old_x, old_y, old_time = \
                        previous_data[track_id]

                    dt = timestamp - old_time

                    if dt > 0:

                        vx = (
                            world_x - old_x
                        ) / dt

                        vy = (
                            world_y - old_y
                        ) / dt

                previous_data[track_id] = (
                    world_x,
                    world_y,
                    timestamp
                )

                # ------------------------------------------------
                # ANGLE
                # ------------------------------------------------

                angle = math.atan2(
                    vy,
                    vx
                )

                # ------------------------------------------------
                # ADD TO HISTORY
                # ------------------------------------------------

                history[track_id].append([
                    world_x,
                    world_y,
                    vx,
                    vy,
                    angle
                ])

                # ------------------------------------------------
                # DRAW DETECTION
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{object_name} ID:{track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (int(pixel_x), int(pixel_y)),
                    5,
                    (0, 0, 255),
                    -1
                )

                # ------------------------------------------------
                # NEED 20 FRAMES
                # ------------------------------------------------

                if len(history[track_id]) < PAST_FRAMES:

                    cv2.putText(
                        frame,
                        f"Collecting: {len(history[track_id])}/20",
                        (int(x1), int(y2) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

                    continue

                # ------------------------------------------------
                # CREATE LSTM INPUT
                # ------------------------------------------------

                features = np.array(
                    history[track_id],
                    dtype=np.float32
                )

                # Normalize using training scaler
                scaled_features = scaler.transform(
                    features
                )

                model_input = scaled_features[
                    np.newaxis,
                    :,
                    :
                ].astype(np.float32)

                # ------------------------------------------------
                # LSTM PREDICTION
                # ------------------------------------------------

                prediction = lstm.predict(
                    model_input,
                    verbose=0
                )[0]

                # ------------------------------------------------
                # NORMALIZED OUTPUT → WORLD COORDINATES
                # ------------------------------------------------

                dummy = np.zeros(
                    (FUTURE_FRAMES, 5),
                    dtype=np.float32
                )

                dummy[:, 0] = prediction[:, 0]
                dummy[:, 1] = prediction[:, 1]

                predicted_world = scaler.inverse_transform(
                    dummy
                )[:, :2]

                # ------------------------------------------------
                # DRAW PAST TRAJECTORY
                # ------------------------------------------------

                past_points = []

                for point in features:

                    wx = point[0]
                    wy = point[1]

                    p = np.float32([
                        [[wx, wy]]
                    ])

                    # World → pixel
                    inverse_H = np.linalg.inv(H)

                    pixel = cv2.perspectiveTransform(
                        p,
                        inverse_H
                    )

                    px = int(pixel[0][0][0])
                    py = int(pixel[0][0][1])

                    past_points.append(
                        (px, py)
                    )

                for i in range(
                    1,
                    len(past_points)
                ):

                    cv2.line(
                        frame,
                        past_points[i - 1],
                        past_points[i],
                        (255, 0, 0),
                        2
                    )

                # ------------------------------------------------
                # DRAW FUTURE PREDICTION
                # ------------------------------------------------

                future_points = []

                for point in predicted_world:

                    wx = point[0]
                    wy = point[1]

                    p = np.float32([
                        [[wx, wy]]
                    ])

                    pixel = cv2.perspectiveTransform(
                        p,
                        inverse_H
                    )

                    px = int(pixel[0][0][0])
                    py = int(pixel[0][0][1])

                    future_points.append(
                        (px, py)
                    )

                for i in range(
                    1,
                    len(future_points)
                ):

                    cv2.line(
                        frame,
                        future_points[i - 1],
                        future_points[i],
                        (0, 0, 255),
                        3
                    )

                    cv2.circle(
                        frame,
                        future_points[i],
                        4,
                        (0, 0, 255),
                        -1
                    )

                # ------------------------------------------------
                # DISPLAY INFO
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    "LSTM Prediction Active",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    # ------------------------------------------------
    # DISPLAY
    # ------------------------------------------------

    cv2.putText(
        frame,
        "Blue = Past | Red = Predicted Future | Q = Exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Trajectory Prediction - Live Camera",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ------------------------------------------------python live_camera_lstm.py
# CLEANUP
# ------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print(" LIVE PREDICTION STOPPED")
print("========================================")