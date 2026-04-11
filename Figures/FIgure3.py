import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("portfolio_monthly_revenue.csv")

# Filter to selected specification
plot_df = df[
    (df["scenario"] == "main_extended") &
    (df["fixed_share_assumption"] == 0.8) &
    (df["stage"] == "test") &
    (df["model"] == "ExtraTrees")
].copy()

# Boxplot
plt.figure(figsize=(7, 5))
plt.boxplot(
    [
        plot_df["actual_revenue_gbp_blended"],
        plot_df["predicted_revenue_gbp_blended"]
    ],
    labels=["Actual", "Predicted"],
    patch_artist=True
)

plt.ylabel("Portfolio revenue (£)")
plt.title("Portfolio revenue distribution: actual vs predicted")

plt.tight_layout()
plt.savefig("revenue_distribution_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()