import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model


# ==========================================
# CONFIGURATION
# ==========================================

YOLO_MODEL = "yolo11n.pt"
LSTM_MODEL = "../models/lstm_trajectory.keras"

PAST_FRAMES = 20
FUTURE_FRAMES = 10

# Object classes
TARGET_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "sports ball"
}


print("========================================")
print(" REAL-TIME TRAJECTORY PREDICTION")
print("========================================")


# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded!")

print("\nLoading LSTM...")

lstm = load_model(LSTM_MODEL)

print("LSTM loaded!")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Camera could not open.")

    exit()


# ==========================================
# TRAJECTORY HISTORY
# ==========================================

history = defaultdict(
    lambda: deque(
        maxlen=PAST_FRAMES
    )
)


frame_id = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_id += 1

    results = yolo.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]


    # ======================================
    # TRACKED OBJECTS
    # ======================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        coordinates = (
            boxes.xyxy
            .cpu()
            .tolist()
        )


        for track_id, class_id, box in zip(
            track_ids,
            classes,
            coordinates
        ):

            object_name = yolo.names[
                class_id
            ]

            if object_name not in TARGET_CLASSES:
                continue


            x1, y1, x2, y2 = box


            # ==================================
            # CENTER POINT
            # ==================================

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2


            history[track_id].append(
                [
                    center_x,
                    center_y
                ]
            )


            # ==================================
            # DRAW OBJECT
            # ==================================

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


            # ==================================
            # DRAW PAST TRAJECTORY
            # ==================================

            points = list(
                history[track_id]
            )

            for i in range(
                1,
                len(points)
            ):

                p1 = points[i - 1]
                p2 = points[i]

                cv2.line(
                    frame,
                    (
                        int(p1[0]),
                        int(p1[1])
                    ),
                    (
                        int(p2[0]),
                        int(p2[1])
                    ),
                    (255, 0, 0),
                    2
                )


            # ==================================
            # PREDICTION
            # ==================================

            if len(
                history[track_id]
            ) >= PAST_FRAMES:

                trajectory = np.array(
                    history[track_id],
                    dtype=np.float32
                )


                # --------------------------------
                # Estimate velocity
                # --------------------------------

                velocity = np.diff(
                    trajectory,
                    axis=0
                )

                velocity = np.vstack(
                    [
                        velocity[0],
                        velocity
                    ]
                )


                # --------------------------------
                # Create 5 features
                #
                # x
                # y
                # vx
                # vy
                # angle
                # --------------------------------

                features = []

                for i in range(
                    len(trajectory)
                ):

                    x = trajectory[i][0]
                    y = trajectory[i][1]

                    vx = velocity[i][0]
                    vy = velocity[i][1]

                    angle = np.arctan2(
                        vy,
                        vx
                    )

                    features.append(
                        [
                            x,
                            y,
                            vx,
                            vy,
                            angle
                        ]
                    )


                features = np.array(
                    features,
                    dtype=np.float32
                )


                # ==================================
                # NOTE
                # ==================================
                # Current LSTM was trained on
                # INTERACTION dataset coordinates.
                #
                # Camera pixel coordinates are NOT
                # directly equivalent.
                #
                # This is therefore a prototype
                # visualization until camera
                # calibration/domain adaptation
                # is added.
                # ==================================


                model_input = features[
                    np.newaxis,
                    ...
                ]


                try:

                    prediction = lstm.predict(
                        model_input,
                        verbose=0
                    )[0]


                    # ==================================
                    # DRAW FUTURE TRAJECTORY
                    # ==================================

                    last_point = trajectory[-1]


                    for predicted_point in prediction:

                        px = predicted_point[0]
                        py = predicted_point[1]


                        cv2.circle(
                            frame,
                            (
                                int(px),
                                int(py)
                            ),
                            4,
                            (0, 0, 255),
                            -1
                        )


                        cv2.line(
                            frame,
                            (
                                int(last_point[0]),
                                int(last_point[1])
                            ),
                            (
                                int(px),
                                int(py)
                            ),
                            (0, 0, 255),
                            2
                        )


                        last_point = predicted_point


                    cv2.putText(
                        frame,
                        "Future trajectory predicted",
                        (
                            int(x1),
                            int(y2) + 25
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )


                except Exception as error:

                    cv2.putText(
                        frame,
                        "Prediction unavailable",
                        (
                            int(x1),
                            int(y2) + 25
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        "REAL-TIME TRAJECTORY PREDICTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Trajectory Prediction",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("\n========================================")
print(" REAL-TIME PREDICTION STOPPED")
print("========================================")