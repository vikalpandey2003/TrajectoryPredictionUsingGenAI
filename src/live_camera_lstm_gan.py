import os
import sys
import time
import cv2
import joblib
import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# PATHS & SETTINGS
# =========================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(*p):
    return os.path.join(BASE, *p)

YOLO_PATH = path("models", "yolo11n.pt")
LSTM_PATH = path("models", "camera_lstm_trajectory.keras")
GAN_PATH = path("models", "camera_trajectory_generator.keras")

SCALER_PATH = path("data", "processed", "camera_scaler.pkl")
LSTM_TARGET_PATH = path("data", "processed", "camera_target_scaler.pkl")
GAN_TARGET_PATH = path("data", "processed", "camera_gan_target_scaler.pkl")

PAST = 20
FUTURE = 10
PREDICT_EVERY = 5
NOISE_DIM = 16
GAN_SAMPLES = 8

WIDTH, HEIGHT = 1280, 720

TARGET_CLASSES = {
    "person", "car", "motorcycle", "bus",
    "truck", "bicycle", "sports ball"
}

# =========================
# HOMOGRAPHY
# =========================
IMAGE_POINTS = np.float32([
    [467, 475],
    [1208, 475],
    [1275, 575],
    [400, 575]
])

WORLD_POINTS = np.float32([
    [0, 0],
    [20, 0],
    [20, 8],
    [0, 8]
])

H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)

if H is None:
    print("ERROR: Homography failed.")
    sys.exit()

H_INV = np.linalg.inv(H)


def transform(x, y, matrix):
    p = np.array([[[x, y]]], dtype=np.float32)
    r = cv2.perspectiveTransform(p, matrix)[0, 0]
    return float(r[0]), float(r[1])


def pixel_to_world(x, y):
    return transform(x, y, H)


def world_to_pixel(x, y):
    x, y = transform(x, y, H_INV)
    return int(x), int(y)


# =========================
# LOAD MODELS
# =========================
print("Loading models...")

yolo = YOLO(YOLO_PATH)
lstm = tf.keras.models.load_model(LSTM_PATH, compile=False)
gan = tf.keras.models.load_model(GAN_PATH, compile=False)

feature_scaler = joblib.load(SCALER_PATH)
lstm_scaler = joblib.load(LSTM_TARGET_PATH)
gan_scaler = joblib.load(GAN_TARGET_PATH)

print("All models loaded successfully.")


# =========================
# DATA STORAGE
# =========================
history = defaultdict(lambda: deque(maxlen=PAST))
previous = {}
lstm_predictions = {}
gan_predictions = {}


# =========================
# TARGET INVERSE SCALING
# =========================
def inverse_target(prediction, scaler):
    minimum = np.asarray(scaler["min"], dtype=np.float32)
    maximum = np.asarray(scaler["max"], dtype=np.float32)

    return prediction * (maximum - minimum) + minimum


# =========================
# PREPARE INPUT
# =========================
def prepare_input(track):
    data = np.asarray(list(track)[-PAST:], dtype=np.float32)
    scaled = feature_scaler.transform(data[:, :5])
    return scaled.reshape(1, PAST, 5)


# =========================
# LSTM PREDICTION
# =========================
def predict_lstm(track):
    if len(track) < PAST:
        return None

    try:
        x = prepare_input(track)
        pred = lstm.predict(x, verbose=0)[0]
        return inverse_target(pred, lstm_scaler)
    except Exception as e:
        print("LSTM error:", e)
        return None


# =========================
# GAN PREDICTION
# =========================
def predict_gan(track):
    if len(track) < PAST:
        return None

    try:
        x = prepare_input(track)
        x = np.repeat(x, GAN_SAMPLES, axis=0)

        noise = np.random.normal(
            0, 1, (GAN_SAMPLES, NOISE_DIM)
        ).astype(np.float32)

        pred = gan.predict([x, noise], verbose=0)
        return inverse_target(pred, gan_scaler)

    except Exception as e:
        print("GAN error:", e)
        return None


# =========================
# DRAW TRAJECTORY
# =========================
def draw_path(frame, points, color, thickness=2, start=None):
    if start is None:
        start = points[0]

    previous_point = world_to_pixel(*start)

    for x, y in points:
        current = world_to_pixel(x, y)
        cv2.line(
            frame,
            previous_point,
            current,
            color,
            thickness
        )
        previous_point = current


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    sys.exit()

print("\n====================================")
print("LIVE CAMERA LSTM + GAN")
print("====================================")
print(f"Past frames      : {PAST}")
print(f"Future frames    : {FUTURE}")
print(f"GAN trajectories : {GAN_SAMPLES}")
print(f"Prediction every : {PREDICT_EVERY} frames")
print("Press Q to exit.\n")


# =========================
# FPS
# =========================
frame_no = 0
fps = 0
fps_count = 0
fps_start = time.time()


# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame could not be read.")
        break

    frame_no += 1
    fps_count += 1

    results = yolo.track(
        frame,
        persist=True,
        verbose=False
    )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        if boxes.id is not None:

            ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(
                coords, ids, classes
            ):

                name = yolo.names[class_id]

                if name not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Bottom-center point
                cx = (x1 + x2) / 2
                cy = y2

                wx, wy = pixel_to_world(cx, cy)

                # Velocity
                now = time.time()
                vx = vy = 0.0

                if track_id in previous:
                    old_x, old_y, old_time = previous[track_id]
                    dt = now - old_time

                    if dt > 0:
                        vx = (wx - old_x) / dt
                        vy = (wy - old_y) / dt

                previous[track_id] = (wx, wy, now)

                speed = np.hypot(vx, vy)

                # [world_x, world_y, vx, vy, speed]
                history[track_id].append([
                    wx, wy, vx, vy, speed
                ])

                track = history[track_id]

                # =========================
                # PREDICTION
                # =========================
                if (
                    len(track) >= PAST
                    and frame_no % PREDICT_EVERY == 0
                ):

                    lstm_pred = predict_lstm(track)
                    gan_pred = predict_gan(track)

                    if lstm_pred is not None:
                        lstm_predictions[track_id] = lstm_pred

                    if gan_pred is not None:
                        gan_predictions[track_id] = gan_pred

                # =========================
                # OBJECT
                # =========================
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{name} ID:{track_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # =========================
                # PAST TRAJECTORY
                # =========================
                if len(track) > 1:
                    past = [(p[0], p[1]) for p in track]
                    draw_path(frame, past, (255, 0, 0), 2)

                start = (track[-1][0], track[-1][1])

                # =========================
                # LSTM FUTURE
                # =========================
                if track_id in lstm_predictions:

                    draw_path(
                        frame,
                        lstm_predictions[track_id],
                        (255, 255, 0),
                        3,
                        start
                    )

                # =========================
                # GAN FUTURES
                # =========================
                if track_id in gan_predictions:

                    candidates = gan_predictions[track_id]

                    for candidate in candidates:
                        draw_path(
                            frame,
                            candidate,
                            (0, 0, 255),
                            1,
                            start
                        )

                    mean_pred = np.mean(
                        candidates,
                        axis=0
                    )

                    draw_path(
                        frame,
                        mean_pred,
                        (0, 0, 255),
                        3,
                        start
                    )

    # =========================
    # FPS
    # =========================
    elapsed = time.time() - fps_start

    if elapsed >= 1:
        fps = fps_count / elapsed
        fps_count = 0
        fps_start = time.time()

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =========================
    # LEGEND
    # =========================
    legend = [
        ("BLUE = Past", (255, 0, 0)),
        ("CYAN = Camera LSTM", (255, 255, 0)),
        ("RED = Camera GAN", (0, 0, 255))
    ]

    for i, (text, color) in enumerate(legend):
        cv2.putText(
            frame,
            text,
            (20, 70 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    cv2.imshow(
        "Trajectory Prediction - Camera LSTM + GAN",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print("\nLive prediction stopped.")