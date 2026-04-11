import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("portfolio_monthly_revenue.csv")
df["date"] = pd.to_datetime(df["date"])

plot_df = df[
    (df["scenario"] == "main_extended") &
    (df["fixed_share_assumption"] == 0.8) &
    (df["model"] == "ExtraTrees") &
    (df["stage"] == "test")
].copy()

plot_df = plot_df.sort_values("date")

plt.figure(figsize=(10, 5))
plt.plot(plot_df["date"], plot_df["actual_revenue_gbp_blended"], linewidth=2, label="Actual")
plt.plot(plot_df["date"], plot_df["predicted_revenue_gbp_blended"], linewidth=2, label="Predicted")

plt.xlabel("Date")
plt.ylabel("Portfolio revenue (£)")
plt.title("Actual vs predicted monthly portfolio revenue")
plt.legend(frameon=False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("revenue_comparison_clean.png", dpi=300, bbox_inches="tight")
plt.show()