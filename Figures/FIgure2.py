import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("MASTER_test_rankings.csv")

# Sort by RMSE (ascending = better first)
df = df.sort_values("RMSE")

# Clean model names (optional but recommended)
df["model"] = df["model"].str.replace("_", " ")

# Highlight baselines vs ML
colors = df["model"].apply(
    lambda x: "orange" if "Baseline" in x else "steelblue"
)

plt.figure(figsize=(10, 5))

plt.barh(df["model"], df["RMSE"], color=colors)

plt.xlabel("RMSE")
plt.title("Forecasting performance comparison (test set)")

plt.tight_layout()
plt.savefig("model_comparison_clean.png", dpi=300, bbox_inches="tight")
plt.show()