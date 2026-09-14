# 🚗 Trajectory Prediction Using GenAI

A computer vision and Generative AI project for **vehicle/object trajectory prediction** using **YOLO, LSTM, and Generative Adversarial Networks (GANs)**.

The system detects and tracks moving objects, extracts their historical trajectories, learns their movement patterns, and predicts their future trajectories using both deterministic and generative deep-learning approaches.

---

## 📌 Overview

Trajectory prediction is the task of predicting the future movement of a moving object based on its previous trajectory.

This project implements an end-to-end trajectory prediction pipeline:

```text
Dataset / Camera
       ↓
Object Detection
       ↓
Object Tracking
       ↓
Trajectory Extraction
       ↓
Feature Engineering
       ↓
Sequence Generation
       ↓
LSTM / GAN
       ↓
Future Trajectory Prediction
       ↓
Visualization & Evaluation
```

The project supports:

- Dataset-based trajectory prediction
- Real-time camera-based trajectory prediction
- LSTM-based deterministic prediction
- GAN-based multi-modal prediction
- YOLO-based object detection and tracking
- Camera-to-world coordinate transformation
- Homography-based camera calibration
- Streamlit-based visualization

### Prediction Configuration

```text
20 Historical Frames → 10 Future Frames
```

---

# ✨ Features

- 🚘 Real-time object detection using YOLO
- 🎯 Object tracking
- 📍 Historical trajectory extraction
- 📈 Velocity and movement feature calculation
- 🧠 LSTM-based trajectory prediction
- 🤖 GAN-based multi-modal trajectory generation
- 📹 Real-time webcam prediction
- 🌍 Camera-to-world coordinate transformation
- 📐 Homography-based camera calibration
- 📊 Trajectory visualization
- 📉 ADE / FDE evaluation
- 🖥️ Streamlit dashboard
- ⚡ Real-time LSTM + GAN prediction
- 💾 Pre-trained models included in the repository

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │   Dataset / Camera   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ YOLO Object Detection│
                    │      & Tracking      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Trajectory Extraction│
                    │  x, y, vx, vy, etc. │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sequence Generation  │
                    │ 20 Past Frames       │
                    │ 10 Future Frames     │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │      LSTM       │       │       GAN       │
        │ Deterministic   │       │ Multiple Future │
        │   Prediction    │       │   Predictions   │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Future Trajectory      │
                  │ Visualization &        │
                  │ Evaluation              │
                  └────────────────────────┘
```

---

# 📹 Real-Time Camera Pipeline

The real-time prediction system follows this pipeline:

```text
                         Webcam
                           │
                           ▼
                         OpenCV
                           │
                           ▼
                YOLO Detection + Tracking
                           │
                           ▼
                  Object Track History
                           │
                           ▼
                   Camera Coordinates
                           │
                           ▼
                Homography Transformation
                           │
                           ▼
                    World Coordinates
                           │
                           ▼
                    20 Frame History
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
                LSTM               GAN
                  │                 │
                  ▼                 ▼
          Single Prediction   Multiple Predictions
                  │                 │
                  └────────┬────────┘
                           ▼
                 Future Trajectory
                    Visualization
```

---

# 🧩 Models

## 1. LSTM Trajectory Prediction

The LSTM model learns temporal dependencies from historical trajectory data.

Trajectory data is sequential, meaning that previous movement can be used to predict future movement.

### Input Features

The dataset-based model uses:

```text
x
y
vx
vy
psi_rad
```

Each input sequence contains:

```text
20 Historical Frames
```

### Output

The model predicts:

```text
10 Future Frames
```

with future:

```text
x
y
```

Conceptually:

```text
Historical Trajectory
        │
        ▼
      LSTM
        │
        ▼
Future Trajectory
```

---

# 🤖 GAN-Based Trajectory Prediction

A Generative Adversarial Network is used to generate **multiple plausible future trajectories**.

Instead of producing only one deterministic future trajectory, the generator uses historical trajectory information along with latent noise.

```text
Past Trajectory
      +
Latent Noise
      │
      ▼
 Generator
      │
      ▼
Future Trajectory
```

The discriminator attempts to distinguish between real and generated trajectories.

```text
                  ┌────────────────┐
                  │ Past Trajectory│
                  └───────┬────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ Generator │
                    └─────┬─────┘
                          │
                          ▼
                 Generated Future
                          │
                          ▼
                   ┌────────────┐
                   │Discriminator│
                   └────────────┘
                      ▲      ▲
                      │      │
                    Real     Fake
                  Trajectory Trajectory
```

---

# 🏗️ GAN Architecture

## Generator

The trajectory generator uses an LSTM-based architecture:

```text
Past Trajectory
      │
      ▼
 LSTM (128)
      │
      ├────── Latent Noise
      │
      ▼
 Concatenation
      │
      ▼
 Repeat for Future Steps
      │
      ▼
 LSTM (128)
      │
      ▼
 TimeDistributed Dense
      │
      ▼
10 Future Positions
```

### Latent Noise

```text
Latent Dimension = 16
```

---

## Discriminator

The discriminator receives a future trajectory and predicts whether it is real or generated.

```text
Future Trajectory
      │
      ▼
 LSTM (128)
      │
      ▼
 Dense (64)
      │
      ▼
 Sigmoid
      │
      ▼
 Real / Fake
```

---

# 📐 Camera Calibration

The real-time camera pipeline uses **homography transformation** to convert image coordinates into world coordinates.

This is important because:

```text
Camera Pixel Coordinates
```

and:

```text
World Coordinates
```

represent different coordinate systems.

The transformation allows the trajectory prediction model to operate on calibrated coordinates.

The live camera implementation contains predefined image and world points for the homography transformation.

---

# 📊 Data Processing

The dataset preprocessing pipeline creates trajectory sequences using:

```text
20 Past Frames
+
10 Future Frames
```

### Dataset Input Features

```text
x
y
vx
vy
psi_rad
```

### Prediction Targets

```text
x
y
```

The preprocessing pipeline applies `MinMaxScaler` for feature scaling.

The scaler is saved and reused during inference.

---

# 🎥 Camera Data Processing

The camera trajectory pipeline extracts information from tracked objects.

The generated trajectory data contains:

```text
track_id
frame_id
timestamp_ms
object_type
x
y
vx
vy
speed
```

For camera-based sequence generation, the features include:

```text
world_x
world_y
vx
vy
speed
```

The prediction targets are:

```text
world_x
world_y
```

---

# 📁 Project Structure

```text
TrajectoryPredictionUsingGenAI/
│
├── app/
│   ├── app.py
│   └── yolo11n.pt
│
├── data/
│   ├── raw/
│   │   └── vehicle_tracks_000.csv
│   │
│   └── processed/
│       ├── calibrated_camera_trajectories.csv
│       ├── camera_gan_target_scaler.pkl
│       ├── camera_lstm_predictions.csv
│       ├── camera_scaler.pkl
│       ├── camera_target_scaler.pkl
│       ├── camera_trajectories.csv
│       ├── camera_trajectory_sequences.npz
│       ├── scaler.pkl
│       └── trajectory_sequences.npz
│
├── models/
│   ├── camera_lstm_trajectory.keras
│   ├── camera_trajectory_discriminator.keras
│   ├── camera_trajectory_generator.keras
│   ├── lstm_trajectory.keras
│   ├── trajectory_discriminator.keras
│   ├── trajectory_generator.keras
│   └── yolo11n.pt
│
├── results/
│   └── graphs/
│
├── src/
│   ├── preprocess.py
│   ├── train_lstm.py
│   ├── train_gan.py
│   ├── camera_tracking.py
│   ├── camera_sequences.py
│   ├── live_camera_lstm.py
│   ├── live_camera_lstm_gan.py
│   ├── realtime_prediction.py
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Requirements

## Software Requirements

Recommended environment:

- Python **3.12.x**
- Git
- Webcam for real-time prediction
- Windows / Linux / macOS

Recommended Python version:

```text
Python 3.12.10
```

---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/codeX4r/TrajectoryPredictionUsingGenAI.git
```

Navigate to the project:

```bash
cd TrajectoryPredictionUsingGenAI
```

Verify the structure:

```text
TrajectoryPredictionUsingGenAI/
├── app/
├── data/
├── models/
├── results/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 2. Create a Virtual Environment

```bash
python -m venv venv
```

This creates:

```text
venv/
```

inside the project directory.

---

# 3. Activate the Virtual Environment

## Windows

```bash
venv\Scripts\activate
```

After activation, you should see:

```text
(venv)
```

Example:

```text
(venv) E:\fp\TrajectoryPredictionUsingGenAI>
```

---

## Linux / macOS

```bash
source venv/bin/activate
```

---

# 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 5. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
tensorflow
opencv-python
ultralytics
streamlit
```

---

# 6. Verify Installation

Run:

```bash
python -c "import tensorflow, cv2, ultralytics, streamlit, numpy, pandas, sklearn, joblib; print('All libraries working')"
```

Expected output:

```text
All libraries working
```

---

# 7. Check TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Recommended version:

```text
TensorFlow 2.21.0
```

---

# 8. Verify Models and Data

The repository contains trained models and processed files.

Check:

```text
models/
data/processed/
```

You should find models such as:

```text
models/
├── lstm_trajectory.keras
├── trajectory_generator.keras
├── trajectory_discriminator.keras
├── camera_lstm_trajectory.keras
├── camera_trajectory_generator.keras
├── camera_trajectory_discriminator.keras
└── yolo11n.pt
```

> **Training is not required to run the existing demo because the trained models are already included in the repository.**

---

# ▶️ Running the Project

## 🖥️ Streamlit Dashboard

From the project root:

```bash
python -m streamlit run app\app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the address in your browser.

---

# 📹 Real-Time LSTM + GAN Prediction

Navigate to the source directory:

```bash
cd src
```

Run:

```bash
python live_camera_lstm_gan.py
```

The system will:

1. Open the webcam
2. Detect objects using YOLO
3. Track detected objects
4. Collect trajectory history
5. Convert camera coordinates into world coordinates
6. Use LSTM to predict future movement
7. Use GAN to generate multiple possible trajectories
8. Visualize predicted trajectories

Press:

```text
Q
```

to exit the camera window.

---

# 🧠 LSTM-Only Real-Time Prediction

From the `src` directory:

```bash
python live_camera_lstm.py
```

This runs the real-time LSTM trajectory prediction pipeline without the GAN generation step.

---

# 🎥 Collect Camera Trajectory Data

To collect trajectory data using the webcam:

```bash
python camera_tracking.py
```

The generated data is saved to:

```text
data/processed/camera_trajectories.csv
```

The collected data contains:

```text
track_id
frame_id
timestamp_ms
object_type
x
y
vx
vy
speed
```

Press:

```text
Q
```

to stop camera recording.

---

# 📐 Generate Camera Sequences

After collecting camera trajectory data:

```bash
python camera_sequences.py
```

This prepares the camera trajectory data for model training/inference.

The camera pipeline uses:

```text
world_x
world_y
vx
vy
speed
```

as input features.

Targets:

```text
world_x
world_y
```

---

# 📊 Dataset Preprocessing

For the dataset-based trajectory pipeline:

```bash
python preprocess.py
```

The preprocessing pipeline creates:

```text
20 Historical Frames
        ↓
10 Future Frames
```

Processed files are stored in:

```text
data/processed/
```

The preprocessing pipeline also generates the scaler required during inference.

---

# 🏋️ Model Training

Pre-trained models are included in the repository.

Therefore, training is **optional** for running the existing demo.

If you want to retrain the models, use the following commands.

---

## Train LSTM

From the `src` directory:

```bash
python train_lstm.py
```

The trained model is stored under:

```text
models/
```

---

## Train GAN

```bash
python train_gan.py
```

The GAN training process consists of:

```text
Generator
     +
Discriminator
```

The trained models are stored under:

```text
models/
```

---

# 🤖 Pre-trained Models

| Model | Purpose |
|---|---|
| `lstm_trajectory.keras` | Dataset-based LSTM prediction |
| `trajectory_generator.keras` | Dataset-based GAN generator |
| `trajectory_discriminator.keras` | Dataset-based GAN discriminator |
| `camera_lstm_trajectory.keras` | Camera-based LSTM prediction |
| `camera_trajectory_generator.keras` | Camera-based GAN generator |
| `camera_trajectory_discriminator.keras` | Camera-based GAN discriminator |
| `yolo11n.pt` | YOLO object detection |

---

# 📈 Evaluation

Trajectory prediction can be evaluated using metrics such as:

## ADE — Average Displacement Error

ADE measures the average displacement between predicted and ground-truth future positions over the prediction horizon.

```text
Lower ADE → Better trajectory prediction
```

---

## FDE — Final Displacement Error

FDE measures the displacement between the predicted final position and the actual final position.

```text
Lower FDE → Better final-position prediction
```

---

# 📊 Results and Visualization

Generated graphs and visualization results are stored under:

```text
results/graphs/
```

These results can be used to inspect:

- Training performance
- Predicted trajectories
- Ground-truth vs predicted trajectories
- GAN-generated trajectories
- Model behavior

---

# 🧪 Prediction Configuration

| Parameter | Value |
|---|---:|
| Historical frames | 20 |
| Future frames | 10 |
| GAN latent dimension | 16 |
| GAN prediction | Multiple trajectories |
| Camera prediction | Real-time |
| Coordinate transformation | Homography |

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | LSTM and GAN models |
| YOLO / Ultralytics | Object detection and tracking |
| OpenCV | Computer vision and camera processing |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Scikit-learn | Data preprocessing and scaling |
| Joblib | Scaler serialization |
| Matplotlib | Visualization |
| Streamlit | Interactive dashboard |

---

# 🔬 Concepts Demonstrated

This project demonstrates practical implementation of:

- Computer Vision
- Object Detection
- Object Tracking
- Trajectory Prediction
- Time-Series Modeling
- LSTM Networks
- Generative Adversarial Networks
- Generative AI
- Multi-modal Prediction
- Feature Engineering
- Data Preprocessing
- Coordinate Transformation
- Homography
- Model Evaluation
- Real-Time Inference

---

# 🚘 Potential Applications

Trajectory prediction can be useful in:

- Autonomous driving
- Advanced Driver Assistance Systems (ADAS)
- Traffic monitoring
- Intelligent transportation systems
- Collision-risk analysis
- Pedestrian movement prediction
- Vehicle behavior analysis
- Robotics
- Smart-city applications

---

# ⚠️ Limitations

The performance of the real-time camera pipeline depends on the quality of camera calibration and object tracking.

Factors that can affect prediction include:

- Camera viewpoint
- Object detection accuracy
- Tracking stability
- Object occlusion
- Lighting conditions
- Object density
- Camera calibration accuracy
- Difference between training and real-world camera data

The basic real-time prediction pipeline should be treated as a prototype when the training data and camera coordinate systems are different.

---

# 🔮 Future Improvements

Possible future improvements include:

- [ ] Improved camera calibration
- [ ] More robust multi-object tracking
- [ ] Transformer-based trajectory prediction
- [ ] Attention-based trajectory prediction
- [ ] Social interaction modeling between vehicles
- [ ] Improved GAN architecture
- [ ] Collision probability prediction
- [ ] Better uncertainty estimation
- [ ] Larger real-world trajectory datasets
- [ ] Model optimization for edge devices
- [ ] Docker deployment
- [ ] Cloud-based inference
- [ ] Improved real-time performance

---

# 🔄 Updating an Existing Clone

If the repository is already cloned and a newer version has been pushed to GitHub, do not clone it again.

Navigate to the project:

```bash
cd TrajectoryPredictionUsingGenAI
```

Pull the latest changes:

```bash
git pull origin main
```

If dependencies have changed:

```bash
pip install -r requirements.txt
```

---

# 🧹 Deactivate the Virtual Environment

When finished working:

```bash
deactivate
```

---

# 🛑 Troubleshooting

## Python Version Problem

Check the installed Python version:

```bash
python --version
```

Recommended:

```text
Python 3.12.x
```

If you are using Python 3.13 and encounter TensorFlow compatibility issues, use Python 3.12.x.

---

## TensorFlow Import Error

Check TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

If TensorFlow cannot be imported, reinstall the dependencies:

```bash
pip install -r requirements.txt
```

---

## YOLO Model Not Found

Make sure the YOLO model exists.

Main model:

```text
models/yolo11n.pt
```

Streamlit application model:

```text
app/yolo11n.pt
```

---

## Webcam Not Opening

Make sure:

- The webcam is connected.
- No other application is using the webcam.
- The correct camera index is configured.
- Camera permissions are enabled.

The default camera index is:

```python
0
```

---

## Streamlit Not Starting

Instead of:

```bash
streamlit run app\app.py
```

use:

```bash
python -m streamlit run app\app.py
```

This ensures Streamlit uses the active Python environment.

---

# ⚡ Complete Windows Setup

For a fresh Windows machine:

```bash
git clone https://github.com/codeX4r/TrajectoryPredictionUsingGenAI.git

cd TrajectoryPredictionUsingGenAI

python -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

python -c "import tensorflow, cv2, ultralytics, streamlit, numpy, pandas, sklearn, joblib; print('All libraries working')"

python -m streamlit run app\app.py
```

For real-time LSTM + GAN:

```bash
cd src

python live_camera_lstm_gan.py
```

---

# 🐧 Complete Linux / macOS Setup

```bash
git clone https://github.com/codeX4r/TrajectoryPredictionUsingGenAI.git

cd TrajectoryPredictionUsingGenAI

python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

python -c "import tensorflow, cv2, ultralytics, streamlit, numpy, pandas, sklearn, joblib; print('All libraries working')"

python -m streamlit run app/app.py
```

For real-time LSTM + GAN:

```bash
cd src

python live_camera_lstm_gan.py
```

---

# 🧪 Reproducibility

For reproducible results, use the recommended environment:

```text
Python 3.12.x
```

Create the environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository contains the processed data, scalers, and trained models required by the existing prediction pipelines.

---

# 📋 Quick Commands Reference

| Task | Command |
|---|---|
| Clone repository | `git clone https://github.com/codeX4r/TrajectoryPredictionUsingGenAI.git` |
| Enter project | `cd TrajectoryPredictionUsingGenAI` |
| Create environment | `python -m venv venv` |
| Activate Windows | `venv\Scripts\activate` |
| Activate Linux/macOS | `source venv/bin/activate` |
| Upgrade pip | `python -m pip install --upgrade pip` |
| Install dependencies | `pip install -r requirements.txt` |
| Verify libraries | `python -c "import tensorflow, cv2, ultralytics, streamlit, numpy, pandas, sklearn, joblib; print('All libraries working')"` |
| Run dashboard | `python -m streamlit run app\app.py` |
| Run LSTM + GAN | `python live_camera_lstm_gan.py` |
| Run LSTM | `python live_camera_lstm.py` |
| Collect camera data | `python camera_tracking.py` |
| Generate camera sequences | `python camera_sequences.py` |
| Preprocess dataset | `python preprocess.py` |
| Train LSTM | `python train_lstm.py` |
| Train GAN | `python train_gan.py` |
| Update repository | `git pull origin main` |
| Deactivate environment | `deactivate` |

---

# 📌 Setup Summary

```text
Install Python 3.12.x
        ↓
Install Git
        ↓
Clone Repository
        ↓
Create Virtual Environment
        ↓
Activate Environment
        ↓
Upgrade pip
        ↓
Install requirements.txt
        ↓
Verify Libraries
        ↓
Run Streamlit / Real-Time Prediction
```

---

# ⚡ Quick Start

For users who only want to run the project:

```bash
git clone https://github.com/codeX4r/TrajectoryPredictionUsingGenAI.git

cd TrajectoryPredictionUsingGenAI

python -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

python -c "import tensorflow, cv2, ultralytics, streamlit, numpy, pandas, sklearn, joblib; print('All libraries working')"

python -m streamlit run app\app.py
```

For real-time LSTM + GAN prediction:

```bash
cd src

python live_camera_lstm_gan.py
```

---

# 🔗 Repository

**GitHub Repository:**

https://github.com/codeX4r/TrajectoryPredictionUsingGenAI

---

# 👨‍💻 Author

**codeX4r**

GitHub:

https://github.com/codeX4r

---

# 📜 License

This project is intended for **educational, research, and demonstration purposes**.

---

# ⭐ Project Summary

This project demonstrates an end-to-end trajectory prediction system by combining:

```text
YOLO
  +
Object Tracking
  +
Trajectory Processing
  +
LSTM
  +
GAN
  +
Camera Calibration
  +
Real-Time Computer Vision
```

The system starts with **object detection and tracking**, extracts historical movement information, processes the trajectory data, and uses deep-learning models to predict possible future movement.

The combination of LSTM and GAN-based prediction provides:

```text
LSTM
  ↓
Deterministic Future Trajectory

GAN
  ↓
Multiple Plausible Future Trajectories
```

This makes the project a practical demonstration of:

**Computer Vision + Deep Learning + Generative AI + Real-Time Trajectory Prediction**