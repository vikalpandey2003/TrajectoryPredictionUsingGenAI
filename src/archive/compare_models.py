import pandas as pd
import matplotlib.pyplot as plt
import os

LSTM_PATH = "../results/metrics/lstm_metrics.csv"
GAN_PATH = "../results/metrics/gan_metrics.csv"

print("========================================")
print(" LSTM vs GAN COMPARISON")
print("========================================")

# Load metrics
lstm = pd.read_csv(LSTM_PATH)
gan = pd.read_csv(GAN_PATH)

# Add evaluation type
lstm["Evaluation"] = "Standard"
gan["Evaluation"] = "Best-of-20"

# Select common columns
lstm = lstm[["Model", "Evaluation", "ADE", "FDE", "RMSE", "MAE"]]
gan = gan[["Model", "Evaluation", "ADE", "FDE", "RMSE", "MAE"]]

# Combine
comparison = pd.concat(
    [lstm, gan],
    ignore_index=True
)

print("\n========================================")
print(" RESULTS")
print("========================================")

print("\n")
print(comparison.to_string(index=False))

# Save comparison
os.makedirs("../results/metrics", exist_ok=True)

comparison.to_csv(
    "../results/metrics/model_comparison.csv",
    index=False
)

print("\nComparison saved at:")
print("../results/metrics/model_comparison.csv")


# ========================================
# ADE / FDE GRAPH
# ========================================

models = comparison["Model"].tolist()

ade_values = comparison["ADE"].tolist()
fde_values = comparison["FDE"].tolist()

plt.figure(figsize=(9, 6))

x = range(len(models))

plt.bar(
    [i - 0.2 for i in x],
    ade_values,
    width=0.4,
    label="ADE"
)

plt.bar(
    [i + 0.2 for i in x],
    fde_values,
    width=0.4,
    label="FDE"
)

plt.xticks(
    list(x),
    models
)

plt.xlabel("Model")
plt.ylabel("Error")
plt.title("LSTM vs GAN - ADE and FDE")

plt.legend()
plt.grid(axis="y")

plt.savefig(
    "../results/graphs/lstm_vs_gan_ade_fde.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ========================================
# RMSE / MAE GRAPH
# ========================================

rmse_values = comparison["RMSE"].tolist()
mae_values = comparison["MAE"].tolist()

plt.figure(figsize=(9, 6))

x = range(len(models))

plt.bar(
    [i - 0.2 for i in x],
    rmse_values,
    width=0.4,
    label="RMSE"
)

plt.bar(
    [i + 0.2 for i in x],
    mae_values,
    width=0.4,
    label="MAE"
)

plt.xticks(
    list(x),
    models
)

plt.xlabel("Model")
plt.ylabel("Error")
plt.title("LSTM vs GAN - RMSE and MAE")

plt.legend()
plt.grid(axis="y")

plt.savefig(
    "../results/graphs/lstm_vs_gan_rmse_mae.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n========================================")
print(" COMPARISON COMPLETED")
print("========================================")

print("\nGraphs saved:")
print("../results/graphs/lstm_vs_gan_ade_fde.png")
print("../results/graphs/lstm_vs_gan_rmse_mae.png")