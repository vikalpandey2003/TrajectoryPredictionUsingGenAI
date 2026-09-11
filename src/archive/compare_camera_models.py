import os
import pandas as pd
import matplotlib.pyplot as plt


# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "graphs"
)


# Input files
lstm_file = os.path.join(
    METRICS_DIR,
    "camera_lstm_metrics.csv"
)

gan_file = os.path.join(
    METRICS_DIR,
    "camera_gan_metrics.csv"
)


# Output file
comparison_file = os.path.join(
    METRICS_DIR,
    "camera_model_comparison.csv"
)


# Load results
lstm = pd.read_csv(lstm_file)
gan = pd.read_csv(gan_file)


# Create comparison table
comparison = pd.DataFrame({
    "Model": [
        "Camera LSTM",
        "Camera GAN"
    ],

    "Evaluation": [
        "Standard",
        "Best-of-20"
    ],

    "ADE": [
        lstm.iloc[0]["ADE"],
        gan.iloc[0]["ADE"]
    ],

    "FDE": [
        lstm.iloc[0]["FDE"],
        gan.iloc[0]["FDE"]
    ],

    "RMSE": [
        lstm.iloc[0]["RMSE"],
        gan.iloc[0]["RMSE"]
    ],

    "MAE": [
        lstm.iloc[0]["MAE"],
        gan.iloc[0]["MAE"]
    ]
})


# Save CSV
comparison.to_csv(
    comparison_file,
    index=False
)


# Display
print("\n========================================")
print("CAMERA MODEL COMPARISON")
print("========================================\n")

print(
    comparison.to_string(
        index=False
    )
)


# ==========================================================
# ADE + FDE GRAPH
# ==========================================================

plt.figure(figsize=(9, 6))

x = range(len(comparison))

plt.bar(
    [i - 0.2 for i in x],
    comparison["ADE"],
    width=0.4,
    label="ADE"
)

plt.bar(
    [i + 0.2 for i in x],
    comparison["FDE"],
    width=0.4,
    label="FDE"
)

plt.xticks(
    list(x),
    comparison["Model"]
)

plt.ylabel("Error")

plt.title(
    "Camera LSTM vs Camera GAN - ADE and FDE"
)

plt.legend()

plt.tight_layout()

ade_fde_file = os.path.join(
    GRAPHS_DIR,
    "camera_lstm_vs_gan_ade_fde.png"
)

plt.savefig(
    ade_fde_file,
    dpi=300
)

plt.close()


# ==========================================================
# RMSE + MAE GRAPH
# ==========================================================

plt.figure(figsize=(9, 6))

plt.bar(
    [i - 0.2 for i in x],
    comparison["RMSE"],
    width=0.4,
    label="RMSE"
)

plt.bar(
    [i + 0.2 for i in x],
    comparison["MAE"],
    width=0.4,
    label="MAE"
)

plt.xticks(
    list(x),
    comparison["Model"]
)

plt.ylabel("Error")

plt.title(
    "Camera LSTM vs Camera GAN - RMSE and MAE"
)

plt.legend()

plt.tight_layout()

rmse_mae_file = os.path.join(
    GRAPHS_DIR,
    "camera_lstm_vs_gan_rmse_mae.png"
)

plt.savefig(
    rmse_mae_file,
    dpi=300
)

plt.close()


# ==========================================================
# BEST MODEL
# ==========================================================

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Best ADE  :",
    comparison.loc[
        comparison["ADE"].idxmin(),
        "Model"
    ]
)

print(
    "Best FDE  :",
    comparison.loc[
        comparison["FDE"].idxmin(),
        "Model"
    ]
)

print(
    "Best RMSE :",
    comparison.loc[
        comparison["RMSE"].idxmin(),
        "Model"
    ]
)

print(
    "Best MAE  :",
    comparison.loc[
        comparison["MAE"].idxmin(),
        "Model"
    ]
)


print("\n========================================")
print("FILES CREATED")
print("========================================")

print(
    "\nCSV:",
    comparison_file
)

print(
    "\nGraph 1:",
    ade_fde_file
)

print(
    "\nGraph 2:",
    rmse_mae_file
)

print(
    "\nComparison completed successfully."
)