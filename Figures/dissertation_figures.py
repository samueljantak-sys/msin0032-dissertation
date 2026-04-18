"""
dissertation_figures.py
=======================
Generates Figures 8-12 for the dissertation. All figures use the main
extended specification; Figures 11 and 12 additionally use the 80% fixed-price
central scenario. Reads pre-computed CSV outputs from model_suite.py and
financial_layer.py — no models are re-estimated.

  Figure 8  — Per-month MAE by model across the 2024 test period
  Figure 9  — Test-set RMSE comparison across all model classes
  Figure 10 — Normalised feature importance (ExtraTrees, main extended spec)
  Figure 11 — Actual vs predicted monthly portfolio revenue
  Figure 12 — Portfolio revenue distribution: actual vs predicted

Outputs are saved to BASE_DIR/dissertation_figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

BASE_DIR          = Path("/Users/samueljantak/Desktop/UCL/Dissertation/final")
MODEL_ROOT        = BASE_DIR / "model_outputs_refined"
FINANCIAL_ROOT    = BASE_DIR / "financial_outputs_refined_multi"
OUT_DIR           = BASE_DIR / "dissertation_figures"
# MAIN_EXTENDED_DIR is the primary results directory. It holds model
# predictions, baseline predictions, and feature importance CSVs for
# the main extended feature specification across all assets.
MAIN_EXTENDED_DIR = MODEL_ROOT / "main_all_assets" / "extended" / "all_assets"

OUT_DIR.mkdir(exist_ok=True)

# A single palette is used across all figures so that the same model or
# feature category always maps to the same colour. The highlight red is
# reserved for the best-performing model and the autoregressive category,
# where temporal persistence is the analytically interesting finding.
COLOURS: dict[str, str] = {
    "baseline":       "#9E9E9E",
    "highlight":      "#E53935",
    "climate_raw":    "#1565C0",
    "lagged":         "#42A5F5",
    "rolling":        "#90CAF9",
    "anomaly":        "#0D47A1",
    "calendar":       "#B0BEC5",
    "autoregressive": "#E53935",
    "interaction":    "#4CAF50",
    "asset_char":     "#78909C",
}

FEATURE_LABEL_MAP: dict[str, str] = {
    "month_sin":   "Month (sine)",
    "month_cos":   "Month (cosine)",
    "month_num":   "Month",
    "year_num":    "Year",
    "quarter_num": "Quarter",
    "winter_flag": "Winter flag",
    "summer_flag": "Summer flag",
    "time_index":  "Time trend",
    "wind_speed_10m":       "Wind speed (10m)",
    "solar_radiation_j_m2": "Solar radiation",
    "temp_2m_c":            "Temperature (2m)",
    "precip_mm":            "Precipitation",
    "wind_u10":             "Wind u-component",
    "wind_v10":             "Wind v-component",
    "wind_speed_10m_lag3":          "Wind speed lag-3",
    "wind_speed_10m_lag6":          "Wind speed lag-6",
    "wind_speed_10m_lag12":         "Wind speed lag-12",
    "solar_radiation_j_m2_lag3":    "Solar radiation lag-3",
    "solar_radiation_j_m2_lag6":    "Solar radiation lag-6",
    "solar_radiation_j_m2_lag12":   "Solar radiation lag-12",
    "temp_2m_c_lag3":               "Temperature lag-3",
    "temp_2m_c_lag6":               "Temperature lag-6",
    "temp_2m_c_lag12":              "Temperature lag-12",
    "precip_mm_lag3":               "Precipitation lag-3",
    "precip_mm_lag6":               "Precipitation lag-6",
    "precip_mm_lag12":              "Precipitation lag-12",
    "wind_speed_10m_roll6_mean":        "Wind speed roll-6 mean",
    "wind_speed_10m_roll12_mean":       "Wind speed roll-12 mean",
    "solar_radiation_j_m2_roll6_mean":  "Solar radiation roll-6 mean",
    "solar_radiation_j_m2_roll12_mean": "Solar radiation roll-12 mean",
    "temp_2m_c_roll6_mean":             "Temperature roll-6 mean",
    "temp_2m_c_roll12_mean":            "Temperature roll-12 mean",
    "precip_mm_roll6_mean":             "Precipitation roll-6 mean",
    "precip_mm_roll12_mean":            "Precipitation roll-12 mean",
    "wind_speed_10m_roll6_std":         "Wind speed roll-6 std",
    "solar_radiation_j_m2_roll6_std":   "Solar radiation roll-6 std",
    "wind_speed_10m_anomaly":           "Wind speed anomaly",
    "solar_radiation_j_m2_anomaly":     "Solar radiation anomaly",
    "temp_2m_c_anomaly":                "Temperature anomaly",
    "precip_mm_anomaly":                "Precipitation anomaly",
    "wind_speed_10m_anomaly_abs":       "Wind speed |anomaly|",
    "solar_radiation_j_m2_anomaly_abs": "Solar radiation |anomaly|",
    "generation_per_mw_lag1":  "Generation lag-1",
    "generation_per_mw_lag2":  "Generation lag-2",
    "generation_per_mw_lag3":  "Generation lag-3",
    "generation_per_mw_lag6":  "Generation lag-6",
    "generation_per_mw_lag12": "Generation lag-12",
    "gen_roll3_mean":           "Generation roll-3 mean",
    "gen_roll6_mean":           "Generation roll-6 mean",
    "gen_roll12_mean":          "Generation roll-12 mean",
    "gen_roll6_std":            "Generation roll-6 std",
    "gen_roll12_std":           "Generation roll-12 std",
    "gen_trend":                "Generation trend",
    "commissioning_year":         "Commissioning year",
    "longitude":                  "Longitude",
    "latitude":                   "Latitude",
    "remaining_asset_life_years": "Remaining asset life",
    "log_solar_radiation": "Log solar radiation",
    "log_wind_speed":      "Log wind speed",
    "solar_radiation_sq":  "Solar radiation\u00b2",
    "wind_speed_sq":       "Wind speed\u00b2",
    "solar_radiation_j_m2 x tech_solar":   "Solar radiation x Solar technology",
    "solar_radiation_j_m2_x_tech_solar":   "Solar radiation x Solar technology",
    "solar_radiation_j_m2__x__tech_solar": "Solar radiation x Solar technology",
    "temp_2m_c x tech_solar":              "Temperature x Solar technology",
    "temp_2m_c_x_tech_solar":              "Temperature x Solar technology",
    "wind_speed_10m x tech_onshore_wind":  "Wind speed x Onshore wind",
    "wind_speed_10m_x_tech_onshore_wind":  "Wind speed x Onshore wind",
    "wind_speed_10m x tech_offshore_wind": "Wind speed x Offshore wind",
    "wind_speed_10m_x_tech_offshore_wind": "Wind speed x Offshore wind",
}


def _strip_pipeline_prefix(feature: str) -> str:
    """Remove the transformer-stage prefix that sklearn prepends to feature names.

    When a ColumnTransformer is fitted, each feature name is prefixed with the
    name of its transformer step (e.g. 'num__month_cos'). These prefixes must
    be stripped before label lookup because FEATURE_LABEL_MAP stores bare names.
    """
    for prefix in (
        "num__", "cat__", "remainder__",
        "pipeline__", "pipeline-1__", "pipeline-2__",
        "columntransformer__", "standardscaler__",
    ):
        if feature.lower().startswith(prefix):
            feature = feature[len(prefix):]
    return feature


def _normalise_interaction(feature: str) -> str:
    """Normalise the separator in interaction term names to a single canonical form.

    Interaction feature names produced by get_dummies and ColumnTransformer use
    different separators (__x__ vs _x_). Normalising to ' x ' before lookup
    ensures all variants resolve to the same display label.
    """
    return feature.replace("__x__", " x ").replace("_x_", " x ")


def get_feature_category(feature: str) -> str:
    """Assign a feature to one of eight analytical categories for colour-coding.

    The category order in the conditional chain reflects priority: interaction
    terms are checked before lags because climate-by-technology products also
    match the raw climate and lag patterns lower in the chain. Autoregressive
    features take precedence over general lags for the same reason.
    """
    f = _normalise_interaction(_strip_pipeline_prefix(feature)).lower()
    if " x " in f or (
        ("tech_" in f or "technology" in f)
        and any(v in f for v in ["solar_radiation", "wind_speed", "temp_2m"])
    ):
        return "interaction"
    if any(x in f for x in ["generation_per_mw", "gen_roll", "gen_lag", "gen_trend"]):
        return "autoregressive"
    if "_anomaly" in f:
        return "anomaly"
    if any(x in f for x in ["_roll", "_std"]) and "anomaly" not in f:
        return "rolling"
    if "_lag" in f:
        return "lagged"
    if any(x in f for x in ["month", "quarter", "winter", "summer", "time_index"]):
        return "calendar"
    if f == "year_num":
        return "calendar"
    if any(x in f for x in ["technology", "country", "tech_"]):
        return "interaction"
    if any(x in f for x in [
        "commissioning", "longitude", "latitude",
        "remaining_asset", "log_solar", "log_wind", "_sq", "year",
    ]):
        return "asset_char"
    return "climate_raw"


def pretty_label(feature: str) -> str:
    """Return a display label for a feature name, with fallback cleaning.

    The lookup runs in three passes: exact match against FEATURE_LABEL_MAP,
    prefix-stripped match, and interaction-normalised match. If all three fail,
    a readable label is constructed from the raw name to avoid cryptic tick
    labels on the importance chart.
    """
    lower_map = {k.lower(): v for k, v in FEATURE_LABEL_MAP.items()}
    for candidate in (
        feature,
        _strip_pipeline_prefix(feature),
        _normalise_interaction(_strip_pipeline_prefix(feature)),
    ):
        if candidate in FEATURE_LABEL_MAP:
            return FEATURE_LABEL_MAP[candidate]
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    stripped = _strip_pipeline_prefix(feature)
    if stripped.startswith("technology_"):
        return f"Technology: {stripped[len('technology_'):].replace('_', ' ').title()}"
    if stripped.startswith("country_"):
        return f"Country: {stripped[len('country_'):].replace('_', ' ').title()}"
    clean = _normalise_interaction(stripped).replace("__", " ").replace("_", " ")
    for pfx in ("num ", "cat ", "remainder "):
        if clean.lower().startswith(pfx):
            clean = clean[len(pfx):]
    return clean.strip().title()


def plot_forecast_error_over_time() -> None:
    """
    Per-month mean absolute error for each model across the 2024 test period,
    corresponding to Figure 8 in the dissertation.

    The model selection logic preserves the same priority order used in the
    results chapter: ExtraTrees first, then other ML models, then the seasonal
    baseline. At most four series are plotted to keep the chart readable.
    Winter (DJF) and summer (JJA) months are shaded because the dissertation
    results show that forecast difficulty is most pronounced in these seasons,
    and the visual annotation helps the reader connect the error pattern to
    the atmospheric persistence argument in Section 8.1.
    """
    pred_file     = MAIN_EXTENDED_DIR / "model_predictions.csv"
    baseline_file = MAIN_EXTENDED_DIR / "baseline_predictions.csv"

    if not pred_file.exists():
        print(f"[skip] {pred_file} not found.")
        return

    preds = pd.read_csv(pred_file)
    if baseline_file.exists():
        preds = pd.concat([preds, pd.read_csv(baseline_file)], ignore_index=True)

    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["stage"] == "test"].copy()

    if preds.empty:
        print("[skip] No test-stage predictions found.")
        return

    preds["abs_error"] = (preds["prediction"] - preds["generation_per_mw"]).abs()
    monthly = (
        preds.groupby(["model", "date"])["abs_error"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    model_priority = [
        "ExtraTrees", "HistGradientBoosting", "RandomForest",
        "Ridge", "ElasticNet", "XGBoost",
        "Baseline_AssetMonthSeasonal", "Baseline_AssetMean",
    ]
    available       = monthly["model"].unique().tolist()
    ordered         = [m for m in model_priority if m in available]
    ordered        += [m for m in available if m not in ordered]
    ml_models       = [m for m in ordered if "Baseline" not in m]
    baseline_models = [m for m in ordered if "Baseline" in m]
    display_models  = (ml_models[:3] + baseline_models[:1]) or ordered[:4]

    blues = ["#1565C0", "#42A5F5", "#90CAF9"]
    style_map: dict[str, dict] = {}
    for i, m in enumerate(display_models):
        if "Baseline" in m:
            style_map[m] = {"color": COLOURS["baseline"], "lw": 1.8, "ls": "--",
                            "zorder": 2, "label": "Seasonal baseline"}
        elif i == 0:
            style_map[m] = {"color": COLOURS["highlight"], "lw": 2.2, "ls": "-",
                            "zorder": 5, "label": m}
        else:
            style_map[m] = {"color": blues[min(i - 1, 2)], "lw": 1.6, "ls": "-",
                            "zorder": 3, "label": m}

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for model in display_models:
        sub = monthly[monthly["model"] == model].sort_values("date")
        if sub.empty:
            continue
        s = style_map.get(model, {"color": "#AAAAAA", "lw": 1.2,
                                  "ls": "-", "zorder": 2, "label": model})
        ax.plot(sub["date"], sub["abs_error"],
                color=s["color"], linewidth=s["lw"], linestyle=s["ls"],
                zorder=s["zorder"], label=s["label"], marker="o", markersize=4)

    dates_in_plot = monthly["date"].sort_values().unique()
    if len(dates_in_plot):
        for d in pd.date_range(dates_in_plot.min(), dates_in_plot.max(), freq="MS"):
            if d.month in [12, 1, 2]:
                ax.axvspan(d, d + pd.offsets.MonthEnd(1),
                           alpha=0.06, color="#5C6BC0", linewidth=0)
            elif d.month in [6, 7, 8]:
                ax.axvspan(d, d + pd.offsets.MonthEnd(1),
                           alpha=0.06, color="#FFA726", linewidth=0)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        mpatches.Patch(color="#5C6BC0", alpha=0.2, label="Winter (DJF)"),
        mpatches.Patch(color="#FFA726", alpha=0.2, label="Summer (JJA)"),
    ]
    labels += ["Winter (DJF)", "Summer (JJA)"]
    ax.legend(handles, labels, loc="upper right", framealpha=0.9,
              edgecolor="#CCCCCC", fontsize=8, ncol=2)

    ax.set_xlabel("Month", labelpad=8)
    ax.set_ylabel("Mean Absolute Error (generation per MW)", labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_title(
        "Per-month forecast error over the 2024 test period (main extended specification)",
        pad=12, fontweight="normal", fontsize=10,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    out_path = OUT_DIR / "figure_08_forecast_error_over_time.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_model_comparison() -> None:
    """
    Horizontal bar chart of test-set RMSE for all models in the main extended
    specification, sorted ascending by error, corresponding to Figure 9.

    Restricting to the main_all_assets scenario matches the primary results
    table in Section 7.1. Baseline models are coloured separately so that the
    performance gap between benchmark and ML approaches is immediately visible.
    """
    rankings_file = MODEL_ROOT / "MASTER_test_rankings.csv"

    if not rankings_file.exists():
        print(f"[skip] {rankings_file} not found.")
        return

    df = pd.read_csv(rankings_file)
    df = df[df["scenario"] == "main_all_assets"].copy()
    df = df.sort_values("RMSE", ascending=True)
    df["model"] = df["model"].str.replace("_", " ")

    colors = df["model"].apply(
        lambda x: "#FFA726" if "Baseline" in x else "#1565C0"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["model"], df["RMSE"], color=colors)
    ax.set_xlabel("RMSE")
    ax.set_title(
        "Forecasting performance comparison (test set, main extended specification)",
        pad=12, fontweight="normal", fontsize=10,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            mpatches.Patch(color="#FFA726", label="Baseline models"),
            mpatches.Patch(color="#1565C0", label="ML models"),
        ],
        framealpha=0.9, edgecolor="#CCCCCC", fontsize=8,
    )

    out_path = OUT_DIR / "figure_09_model_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_feature_importance() -> None:
    """
    Normalised feature importance for the ExtraTrees model under the main
    extended specification, corresponding to Figure 10 in the dissertation.

    ExtraTrees is used as the reference model because it achieves the lowest
    RMSE in the main extended specification. Importance scores are normalised
    to sum to one so that the chart conveys relative contribution rather than
    raw split-count magnitudes, which vary with tree depth and n_estimators.
    The top 15 features are shown to keep the chart legible while retaining
    the full set of analytically meaningful variables discussed in Section 7.2.
    Absolute coefficient magnitudes are used for linear models to allow a
    consistent importance ranking across model types.
    """
    candidates = [
        MAIN_EXTENDED_DIR / "importance_ExtraTrees.csv",
        MAIN_EXTENDED_DIR / "importance_RandomForest.csv",
        MAIN_EXTENDED_DIR / "importance_HistGradientBoosting.csv",
        MAIN_EXTENDED_DIR / "importance_XGBoost.csv",
    ]
    imp_file = next((f for f in candidates if f.exists()), None)

    if imp_file is None:
        all_imp = list(MAIN_EXTENDED_DIR.glob("importance_*.csv"))
        if not all_imp:
            print(f"[skip] No importance files found in {MAIN_EXTENDED_DIR}.")
            return
        imp_file = all_imp[0]

    model_name = imp_file.stem.replace("importance_", "")
    imp = pd.read_csv(imp_file)

    # Absolute coefficients are used for linear models to allow a consistent
    # magnitude-based ranking across model types.
    if "importance" in imp.columns:
        imp = imp.rename(columns={"importance": "value"})
    elif "abs_coefficient" in imp.columns:
        imp = imp.rename(columns={"abs_coefficient": "value"})
    elif "coefficient" in imp.columns:
        imp["value"] = imp["coefficient"].abs()
    else:
        num_cols = imp.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            print("[skip] Could not identify importance values.")
            return
        imp = imp.rename(columns={num_cols[0]: "value"})

    feature_col = "feature" if "feature" in imp.columns else imp.columns[0]
    imp = imp[[feature_col, "value"]].copy()
    imp.columns = ["feature", "value"]
    imp["value"] = pd.to_numeric(imp["value"], errors="coerce").abs()
    imp = imp.dropna(subset=["value"])

    total = imp["value"].sum()
    if total > 0:
        imp["value"] /= total

    imp = imp.sort_values("value", ascending=False).head(15).reset_index(drop=True)
    imp["category"] = imp["feature"].apply(get_feature_category)
    imp["label"]    = imp["feature"].apply(pretty_label)
    imp = imp.sort_values("value", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    bar_colours = [COLOURS[c] for c in imp["category"]]
    bars = ax.barh(imp["label"], imp["value"], color=bar_colours,
                   edgecolor="white", linewidth=0.5, height=0.7)

    for bar, val in zip(bars, imp["value"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color="#444444")

    shown_cats = imp["category"].unique()
    cat_labels = {
        "climate_raw":    "Raw climate variables",
        "lagged":         "Lagged climate variables",
        "rolling":        "Rolling statistics",
        "anomaly":        "Climate anomalies",
        "calendar":       "Calendar / temporal",
        "autoregressive": "Autoregressive (generation lags)",
        "interaction":    "Climate x technology interactions",
        "asset_char":     "Asset characteristics",
    }
    legend_patches = [
        mpatches.Patch(color=COLOURS[c], label=cat_labels[c])
        for c in ["climate_raw", "lagged", "rolling", "anomaly",
                  "calendar", "autoregressive", "interaction", "asset_char"]
        if c in shown_cats
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.9, edgecolor="#CCCCCC", fontsize=8)

    ax.set_xlabel("Normalised feature importance", labelpad=8)
    ax.set_xlim(0, imp["value"].max() * 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_title(
        f"Feature importance — {model_name} model "
        "(main extended specification, test set)",
        pad=12, fontweight="normal", fontsize=10,
    )

    out_path = OUT_DIR / "figure_10_feature_importance.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_revenue_comparison() -> None:
    """
    Line chart of actual and predicted monthly portfolio revenue for the
    ExtraTrees model under the central revenue scenario, corresponding to
    Figure 11 in the dissertation.

    The 80% fixed-price assumption is used as the central scenario because it
    represents the typical contracted revenue share for the ORIT portfolio.
    ExtraTrees is shown as the primary ML comparator because it achieves the
    lowest generation RMSE in the main extended specification.
    """
    revenue_file = FINANCIAL_ROOT / "portfolio_monthly_revenue.csv"

    if not revenue_file.exists():
        print(f"[skip] {revenue_file} not found.")
        return

    df = pd.read_csv(revenue_file)
    df["date"] = pd.to_datetime(df["date"])

    plot_df = df[
        (df["scenario"]              == "main_extended") &
        (df["fixed_share_assumption"] == 0.8) &
        (df["model"]                 == "ExtraTrees") &
        (df["stage"]                 == "test")
    ].sort_values("date")

    if plot_df.empty:
        print("[skip] No matching rows for revenue comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["date"], plot_df["actual_revenue_gbp_blended"],
            linewidth=2, label="Actual", color="#1565C0")
    ax.plot(plot_df["date"], plot_df["predicted_revenue_gbp_blended"],
            linewidth=2, label="Predicted", color=COLOURS["highlight"])

    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Portfolio revenue (£)", labelpad=8)
    ax.set_title(
        "Actual vs predicted monthly portfolio revenue "
        "(ExtraTrees, 80% fixed-price, 2024 test set)",
        pad=12, fontweight="normal", fontsize=10,
    )
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    plt.xticks(rotation=45)

    out_path = OUT_DIR / "figure_11_revenue_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_revenue_distribution() -> None:
    """
    Box plot comparing the distributions of actual and predicted monthly
    portfolio revenue for ExtraTrees, corresponding to Figure 12.

    The compression of the predicted distribution relative to realised outcomes
    is the primary visual evidence for the systematic tail-risk underestimation
    discussed in Section 7.4. The interquartile range and whisker extents are
    the most informative features of this chart: a well-calibrated model would
    show comparable box widths, whereas the narrower predicted box confirms
    that even the best ML model underestimates downside dispersion.
    """
    revenue_file = FINANCIAL_ROOT / "portfolio_monthly_revenue.csv"

    if not revenue_file.exists():
        print(f"[skip] {revenue_file} not found.")
        return

    df = pd.read_csv(revenue_file)
    df["date"] = pd.to_datetime(df["date"])

    plot_df = df[
        (df["scenario"]              == "main_extended") &
        (df["fixed_share_assumption"] == 0.8) &
        (df["model"]                 == "ExtraTrees") &
        (df["stage"]                 == "test")
    ].copy()

    if plot_df.empty:
        print("[skip] No matching rows for revenue distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [
            plot_df["actual_revenue_gbp_blended"].dropna(),
            plot_df["predicted_revenue_gbp_blended"].dropna(),
        ],
        labels=["Actual", "Predicted"],
        patch_artist=True,
        medianprops={"color": "#1565C0", "linewidth": 2},
    )
    for patch, colour in zip(bp["boxes"], ["#BBDEFB", "#FFCCBC"]):
        patch.set_facecolor(colour)

    ax.set_ylabel("Portfolio revenue (£)", labelpad=8)
    ax.set_title(
        "Portfolio revenue distribution: actual vs predicted "
        "(ExtraTrees, 80% fixed-price, 2024 test set)",
        pad=12, fontweight="normal", fontsize=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)

    out_path = OUT_DIR / "figure_12_revenue_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_forecast_error_over_time()
    plot_model_comparison()
    plot_feature_importance()
    plot_revenue_comparison()
    plot_revenue_distribution()
