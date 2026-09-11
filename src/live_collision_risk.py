import cv2
import numpy as np
import math
import joblib

from collections import defaultdict, deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model


# ==================================================
# SETTINGS
# ==================================================

YOLO_MODEL = "yolo11n.pt"
LSTM_MODEL = "../models/lstm_trajectory.keras"
SCALER_PATH = "../data/processed/scaler.pkl"

PAST_FRAMES = 20
FUTURE_FRAMES = 10

# Assumed future time between predictions
FUTURE_DT = 0.1

# Demo distance thresholds
HIGH_RISK_DISTANCE = 1.0
MEDIUM_RISK_DISTANCE = 2.5

TARGET_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "sports ball"
}


# ==================================================
# LOAD MODELS
# ==================================================

print("========================================")
print(" LIVE COLLISION RISK PREDICTION")
print("========================================")

print("\nLoading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading LSTM...")
lstm = load_model(LSTM_MODEL)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("\nModels loaded successfully!")


# ==================================================
# CAMERA CALIBRATION
# ==================================================

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

inverse_H = np.linalg.inv(H)


# ==================================================
# HISTORY
# ==================================================

history = defaultdict(
    lambda: deque(maxlen=PAST_FRAMES)
)

previous_data = {}


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def pixel_to_world(px, py):

    point = np.float32([
        [[px, py]]
    ])

    world = cv2.perspectiveTransform(
        point,
        H
    )

    return (
        float(world[0][0][0]),
        float(world[0][0][1])
    )


def world_to_pixel(x, y):

    point = np.float32([
        [[x, y]]
    ])

    pixel = cv2.perspectiveTransform(
        point,
        inverse_H
    )

    return (
        int(pixel[0][0][0]),
        int(pixel[0][0][1])
    )


def predict_trajectory(track_id):

    if len(history[track_id]) < PAST_FRAMES:
        return None

    features = np.array(
        history[track_id],
        dtype=np.float32
    )

    # Normalize exactly like training data
    scaled_features = scaler.transform(
        features
    )

    model_input = scaled_features[
        np.newaxis,
        :,
        :
    ].astype(np.float32)

    prediction = lstm.predict(
        model_input,
        verbose=0
    )[0]

    # Model output is normalized.
    # Convert X/Y back to original coordinate scale.

    dummy = np.zeros(
        (FUTURE_FRAMES, 5),
        dtype=np.float32
    )

    dummy[:, 0] = prediction[:, 0]
    dummy[:, 1] = prediction[:, 1]

    world_prediction = scaler.inverse_transform(
        dummy
    )[:, :2]

    return world_prediction


def calculate_collision_risk(
    trajectory_a,
    trajectory_b
):

    minimum_distance = float("inf")
    closest_step = -1

    for i in range(
        min(
            len(trajectory_a),
            len(trajectory_b)
        )
    ):

        distance = np.sqrt(
            (trajectory_a[i, 0] - trajectory_b[i, 0]) ** 2
            +
            (trajectory_a[i, 1] - trajectory_b[i, 1]) ** 2
        )

        if distance < minimum_distance:

            minimum_distance = distance
            closest_step = i

    if minimum_distance <= HIGH_RISK_DISTANCE:
        risk = "HIGH"

    elif minimum_distance <= MEDIUM_RISK_DISTANCE:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    if closest_step >= 0:
        time_to_closest = (
            closest_step + 1
        ) * FUTURE_DT
    else:
        time_to_closest = None

    return (
        minimum_distance,
        time_to_closest,
        risk
    )


# ==================================================
# CAMERA
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nCamera open nahi hua.")
    exit()

print("\n========================================")
print(" CAMERA STARTED")
print(" Move two objects in the scene")
print(" Press Q to exit")
print("========================================")


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    current_time = (
        cv2.getTickCount()
        /
        cv2.getTickFrequency()
    )

    results = yolo.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    current_objects = {}

    # ==================================================
    # DETECTION + TRACKING
    # ==================================================

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

                # Bottom-center point
                pixel_x = (x1 + x2) / 2
                pixel_y = y2

                # ==================================================
                # PIXEL -> WORLD
                # ==================================================

                world_x, world_y = pixel_to_world(
                    pixel_x,
                    pixel_y
                )

                # ==================================================
                # VELOCITY
                # ==================================================

                vx = 0.0
                vy = 0.0

                if track_id in previous_data:

                    old_x, old_y, old_time = \
                        previous_data[track_id]

                    dt = current_time - old_time

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
                    current_time
                )

                # ==================================================
                # ANGLE
                # ==================================================

                angle = math.atan2(
                    vy,
                    vx
                )

                # ==================================================
                # SAVE HISTORY
                # ==================================================

                history[track_id].append([
                    world_x,
                    world_y,
                    vx,
                    vy,
                    angle
                ])

                current_objects[track_id] = {
                    "name": object_name,
                    "box": (x1, y1, x2, y2)
                }

                # ==================================================
                # DRAW OBJECT
                # ==================================================

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


    # ==================================================
    # PREDICT TRAJECTORIES
    # ==================================================

    predictions = {}

    for track_id in current_objects:

        prediction = predict_trajectory(
            track_id
        )

        if prediction is not None:

            predictions[track_id] = prediction


    # ==================================================
    # DRAW INDIVIDUAL PREDICTIONS
    # ==================================================

    for track_id, trajectory in predictions.items():

        points = []

        for point in trajectory:

            px, py = world_to_pixel(
                point[0],
                point[1]
            )

            points.append(
                (px, py)
            )

        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 0, 255),
                2
            )


    # ==================================================
    # PAIRWISE COLLISION CHECK
    # ==================================================

    object_ids = list(
        predictions.keys()
    )

    risk_found = False

    for i in range(
        len(object_ids)
    ):

        for j in range(
            i + 1,
            len(object_ids)
        ):

            id_a = object_ids[i]
            id_b = object_ids[j]

            trajectory_a = predictions[id_a]
            trajectory_b = predictions[id_b]

            (
                minimum_distance,
                time_to_closest,
                risk
            ) = calculate_collision_risk(
                trajectory_a,
                trajectory_b
            )

            # ==================================================
            # DISPLAY RISK
            # ==================================================

            if risk == "HIGH":

                risk_found = True

                cv2.putText(
                    frame,
                    f"!!! HIGH RISK !!!",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            elif risk == "MEDIUM":

                if not risk_found:

                    cv2.putText(
                        frame,
                        "MEDIUM RISK",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 165, 255),
                        3
                    )

            else:

                if not risk_found:

                    cv2.putText(
                        frame,
                        "LOW RISK",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        3
                    )

            # ==================================================
            # DETAILS
            # ==================================================

            text = (
                f"ID {id_a} vs ID {id_b} | "
                f"Distance: {minimum_distance:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            if time_to_closest is not None:

                time_text = (
                    f"Closest approach: "
                    f"{time_to_closest:.2f}s"
                )

                cv2.putText(
                    frame,
                    time_text,
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


    # ==================================================
    # HEADER
    # ==================================================

    cv2.putText(
        frame,
        "LSTM Trajectory + Collision Risk",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "RED = Predicted Future | Q = Exit",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Trajectory Collision Risk",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print(" COLLISION RISK SYSTEM STOPPED")
print("========================================")