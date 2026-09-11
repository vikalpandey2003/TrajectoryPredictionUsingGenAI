import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import sys


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Trajectory Prediction Using Generative AI",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

GRAPHS_DIR = os.path.join(
    RESULTS_DIR,
    "graphs"
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def graph_path(filename):

    return os.path.join(
        GRAPHS_DIR,
        filename
    )


def metric_path(filename):

    return os.path.join(
        METRICS_DIR,
        filename
    )


def data_path(filename):

    return os.path.join(
        DATA_DIR,
        filename
    )


def show_image(
    filename,
    caption=None
):

    path = graph_path(filename)

    if os.path.exists(path):

        st.image(
            path,
            caption=caption,
            use_container_width=True
        )

    else:

        st.warning(
            f"Graph not found: {filename}"
        )


def load_csv(filename):

    path = metric_path(filename)

    if os.path.exists(path):

        return pd.read_csv(path)

    return None


# ============================================================
# TITLE
# ============================================================

st.title(
    "Trajectory Prediction Using Generative AI"
)

st.caption(
    "LSTM + GAN based trajectory prediction with YOLO camera integration"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Project Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "EDA",
        "Model Results",
        "Camera Tracking",
        "Trajectory Visualization",
        "Live Camera"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header(
        "Project Overview"
    )

    st.write(
        """
        This project predicts the future trajectory of moving objects
        using deep learning models. The system combines an LSTM based
        trajectory predictor with a GAN based model capable of generating
        multiple possible future trajectories.
        """
    )


    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
Camera / Dataset
       ↓
Object Detection & Tracking
       ↓
Trajectory Data
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
20 Past Frames
       ↓
 ┌───────────────┐
 │               │
 ▼               ▼
LSTM             GAN
 │               │
 ▼               ▼
Single Future   Multiple Possible
Trajectory       Futures
 └───────┬───────┘
         ↓
Trajectory Evaluation
         ↓
Visualization / Dashboard
        """,
        language="text"
    )


    # --------------------------------------------------------
    # Project Statistics
    # --------------------------------------------------------

    st.subheader(
        "Project Statistics"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Past Frames",
            "20"
        )


    with col2:

        st.metric(
            "Future Frames",
            "10"
        )


    with col3:

        st.metric(
            "Camera Objects",
            "16"
        )


    with col4:

        st.metric(
            "Camera Records",
            "2401"
        )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    st.subheader(
        "Models Used"
    )

    model_data = pd.DataFrame({

        "Model": [
            "LSTM",
            "GAN",
            "YOLO"
        ],

        "Purpose": [
            "Future trajectory prediction",
            "Multiple possible future trajectories",
            "Object detection and tracking"
        ]

    })

    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EDA
# ============================================================

elif page == "EDA":

    st.header(
        "Exploratory Data Analysis"
    )

    st.write(
        """
        The dataset was analyzed to understand trajectory patterns,
        object movement, velocity and spatial behavior.
        """
    )


    # --------------------------------------------------------
    # Vehicle trajectory
    # --------------------------------------------------------

    st.subheader(
        "Vehicle Trajectory"
    )

    show_image(
        "vehicle_trajectory.png",
        "Example trajectory from the dataset"
    )


    # --------------------------------------------------------
    # Multiple trajectories
    # --------------------------------------------------------

    st.subheader(
        "Multiple Object Trajectories"
    )

    show_image(
        "multiple_vehicle_trajectories.png",
        "Multiple observed trajectories"
    )


    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    st.subheader(
        "Speed Distribution"
    )

    show_image(
        "speed_distribution.png",
        "Distribution of object speeds"
    )


# ============================================================
# MODEL RESULTS
# ============================================================

elif page == "Model Results":

    st.header(
        "Model Results"
    )


    # ========================================================
    # ORIGINAL DATASET MODELS
    # ========================================================

    st.subheader(
        "Original Dataset Models"
    )


    lstm_df = load_csv(
        "lstm_metrics.csv"
    )

    gan_df = load_csv(
        "gan_metrics.csv"
    )

    comparison_df = load_csv(
        "model_comparison.csv"
    )


    if comparison_df is not None:

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Original model comparison file not found."
        )


    # --------------------------------------------------------
    # Original LSTM
    # --------------------------------------------------------

    st.subheader(
        "LSTM Training"
    )

    show_image(
        "lstm_training_loss.png",
        "LSTM training loss"
    )


    show_image(
        "lstm_prediction.png",
        "LSTM trajectory prediction"
    )


    # --------------------------------------------------------
    # Original GAN
    # --------------------------------------------------------

    st.subheader(
        "GAN Training"
    )

    show_image(
        "gan_training_loss.png",
        "GAN generator and discriminator losses"
    )


    show_image(
        "gan_generated_trajectory.png",
        "GAN generated trajectory"
    )


    # ========================================================
    # CAMERA MODELS
    # ========================================================

    st.divider()

    st.subheader(
        "Camera-Trained Models"
    )


    camera_lstm_df = load_csv(
        "camera_lstm_metrics.csv"
    )

    camera_gan_df = load_csv(
        "camera_gan_metrics.csv"
    )

    camera_comparison_df = load_csv(
        "camera_model_comparison.csv"
    )


    if camera_comparison_df is not None:

        st.markdown(
            "### Camera LSTM vs Camera GAN"
        )

        st.dataframe(
            camera_comparison_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # Metrics cards
    # --------------------------------------------------------

    if (
        camera_lstm_df is not None
        and camera_gan_df is not None
    ):

        lstm_row = camera_lstm_df.iloc[0]

        gan_row = camera_gan_df.iloc[0]


        st.markdown(
            "### Camera LSTM"
        )

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "ADE",
                f"{lstm_row['ADE']:.6f}"
            )


        with c2:

            st.metric(
                "FDE",
                f"{lstm_row['FDE']:.6f}"
            )


        with c3:

            st.metric(
                "RMSE",
                f"{lstm_row['RMSE']:.6f}"
            )


        with c4:

            st.metric(
                "MAE",
                f"{lstm_row['MAE']:.6f}"
            )


        st.markdown(
            "### Camera GAN — Best-of-20"
        )

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "ADE",
                f"{gan_row['ADE']:.6f}"
            )


        with c2:

            st.metric(
                "FDE",
                f"{gan_row['FDE']:.6f}"
            )


        with c3:

            st.metric(
                "RMSE",
                f"{gan_row['RMSE']:.6f}"
            )


        with c4:

            st.metric(
                "MAE",
                f"{gan_row['MAE']:.6f}"
            )


    # --------------------------------------------------------
    # Camera graphs
    # --------------------------------------------------------

    st.subheader(
        "Camera LSTM Training"
    )

    show_image(
        "camera_lstm_fine_tuning.png",
        "Camera LSTM fine-tuning"
    )


    st.subheader(
        "Camera GAN Training"
    )

    show_image(
        "camera_gan_training_loss.png",
        "Camera GAN training loss"
    )


    show_image(
        "camera_gan_generated_trajectory.png",
        "Camera GAN generated trajectories"
    )


    st.subheader(
        "Camera Model Comparison"
    )

    show_image(
        "camera_lstm_vs_gan_ade_fde.png",
        "Camera LSTM vs GAN — ADE and FDE"
    )

    show_image(
        "camera_lstm_vs_gan_rmse_mae.png",
        "Camera LSTM vs GAN — RMSE and MAE"
    )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.info(
        """
        Current camera test results show lower error for the Camera LSTM
        than the Camera GAN on ADE, FDE, RMSE and MAE.

        The GAN is retained because its main advantage is generating
        multiple possible future trajectories rather than only one
        deterministic prediction.

        The reported camera errors are in the project's calibrated
        world-coordinate representation. The current homography uses
        approximate demo dimensions, so these values should not be
        presented as exact physical meters.
        """
    )


# ============================================================
# CAMERA TRACKING
# ============================================================

elif page == "Camera Tracking":

    st.header(
        "Camera Tracking"
    )

    camera_file = data_path(
        "camera_trajectories.csv"
    )


    if os.path.exists(camera_file):

        camera_df = pd.read_csv(
            camera_file
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Records",
                len(camera_df)
            )


        with col2:

            if "track_id" in camera_df.columns:

                st.metric(
                    "Tracked Objects",
                    camera_df["track_id"].nunique()
                )

            else:

                st.metric(
                    "Tracked Objects",
                    "N/A"
                )


        with col3:

            if "agent_type" in camera_df.columns:

                st.metric(
                    "Object Types",
                    camera_df["agent_type"].nunique()
                )

            else:

                st.metric(
                    "Object Types",
                    "N/A"
                )


        with col4:

            if "speed" in camera_df.columns:

                st.metric(
                    "Average Speed",
                    f"{camera_df['speed'].mean():.3f}"
                )

            else:

                st.metric(
                    "Average Speed",
                    "N/A"
                )


        st.subheader(
            "Camera Trajectory Data"
        )

        st.dataframe(
            camera_df.head(100),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Object type distribution
        # ----------------------------------------------------

        if "agent_type" in camera_df.columns:

            st.subheader(
                "Object Type Distribution"
            )

            type_counts = (
                camera_df[
                    "agent_type"
                ]
                .value_counts()
            )

            st.bar_chart(
                type_counts
            )


    else:

        st.warning(
            "camera_trajectories.csv was not found."
        )


# ============================================================
# TRAJECTORY VISUALIZATION
# ============================================================

elif page == "Trajectory Visualization":

    st.header(
        "Trajectory Visualization"
    )

    st.write(
        """
        Comparison between historical movement, actual future trajectory,
        LSTM prediction and the best GAN generated trajectory.
        """
    )


    show_image(
        "final_trajectory_comparison.png",
        "Final trajectory comparison"
    )


    st.subheader(
        "Legend"
    )

    st.markdown(
        """
        🔵 **Historical Trajectory**

        🟠 **Actual Future**

        🟢 **LSTM Prediction**

        🔴 **Best GAN Prediction**
        """
    )


    st.info(
        """
        The visualization is generated from the model evaluation pipeline.
        It is intended to show how the predicted trajectories compare with
        the observed future trajectory.
        """
    )


# ============================================================
# LIVE CAMERA
# ============================================================

elif page == "Live Camera":

    st.header(
        "Live Camera Prediction"
    )

    st.write(
        """
        This module launches the native OpenCV camera application.

        YOLO detects and tracks objects, while the camera-trained LSTM
        and GAN models predict their future trajectories.
        """
    )


    st.subheader(
        "Pipeline"
    )

    st.code(
        """
Live Camera
     ↓
YOLO Detection + Tracking
     ↓
Object Center
     ↓
Pixel → World Coordinates
     ↓
Velocity + Speed
     ↓
20-frame History
     ↓
Camera LSTM + Camera GAN
     ↓
Future Trajectory
        """,
        language="text"
    )


    st.warning(
        """
        Keep the camera in the same approximate scene used for the
        current homography calibration. If the camera position or
        viewpoint changes significantly, recalibration is required.
        """
    )


    if st.button(
        "Start Live Camera",
        type="primary"
    ):

        live_script = os.path.join(
            SRC_DIR,
            "live_camera_lstm_gan.py"
        )


        if os.path.exists(live_script):

            try:

                subprocess.Popen(
                    [
                        sys.executable,
                        live_script
                    ],
                    cwd=SRC_DIR
                )


                st.success(
                    "Live camera started in a separate OpenCV window."
                )

            except Exception as e:

                st.error(
                    f"Could not start live camera: {e}"
                )

        else:

            st.error(
                "live_camera_lstm_gan.py was not found."
            )


    st.markdown(
        """
        ### Controls

        - Press **Q** in the camera window to stop.
        - Green box = YOLO tracked object
        - Blue = observed/past trajectory
        - Cyan = Camera LSTM prediction
        - Red = Camera GAN possible futures
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Trajectory Prediction Using Generative AI"
)

st.sidebar.caption(
    "LSTM + GAN + YOLO"
)