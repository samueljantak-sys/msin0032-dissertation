from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(".")
PANEL_FILE = BASE_DIR / "model_panel_clean.csv"
MODEL_OUTPUT_ROOT = BASE_DIR / "model_outputs_refined"
OUTPUT_DIR = BASE_DIR / "financial_outputs_refined_multi"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "generation_per_mw"
DATE_COL = "date"
ASSET_COL = "asset_id"
TECH_COL = "technology"
PRICE_COL = "power_price_gbp_mwh"

SCENARIOS = {
    "main_extended": {
        "dir": MODEL_OUTPUT_ROOT / "main_all_assets" / "extended" / "all_assets",
        "models": [
            "Baseline_AssetMonthSeasonal",
            "Ridge",
            "ElasticNet",
            "XGBoost",
            "ExtraTrees",
        ],
    },
    "autoreg_robustness": {
        "dir": MODEL_OUTPUT_ROOT / "main_autoreg_features" / "autoreg" / "all_assets",
        "models": [
            "Baseline_AssetMonthSeasonal",
            "ElasticNet",
            "HistGradientBoosting",
            "XGBoost",
            "Ensemble_AvgTop3",
        ],
    },
}

INCLUDE_VALIDATION = True
FIXED_SHARE_CASES = [0.70, 0.80, 0.90]


def clean_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].astype(str)
            if s.str.contains("%", regex=False).any():
                df[col] = (
                    s.str.replace("%", "", regex=False)
                    .str.strip()
                    .replace({"": np.nan, "nan": np.nan})
                    .astype(float)
                    / 100.0
                )
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mean_bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def corr_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_series(y_true, y_pred) -> dict:
    return {
        "MAE": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "sMAPE_pct": smape(y_true, y_pred),
        "Mean_Bias": mean_bias(y_true, y_pred),
        "Corr": corr_safe(y_true, y_pred),
    }


def load_prediction_tables(scenario_dir: Path) -> pd.DataFrame:
    tables = []
    for fname in ["model_predictions.csv", "baseline_predictions.csv", "ensemble_predictions.csv"]:
        f = scenario_dir / fname
        if f.exists():
            tables.append(pd.read_csv(f))

    if not tables:
        raise FileNotFoundError(f"No prediction files found in {scenario_dir}")

    preds = pd.concat(tables, ignore_index=True)
    preds[DATE_COL] = pd.to_datetime(preds[DATE_COL])
    return preds


def find_owned_capacity(df: pd.DataFrame) -> pd.Series:
    if "owned_capacity_mw" in df.columns:
        return df["owned_capacity_mw"]
    if "capacity_mw" in df.columns and "ownership_fraction" in df.columns:
        return df["capacity_mw"] * df["ownership_fraction"]
    if "capacity_mw" in df.columns and "ownership_pct" in df.columns:
        return df["capacity_mw"] * df["ownership_pct"]
    raise ValueError("Need capacity and ownership to construct owned capacity.")


def risk_summary(s: pd.Series) -> dict:
    s = s.dropna()
    mean_val = float(s.mean())
    std_val = float(s.std())
    return {
        "mean": mean_val,
        "std_dev": std_val,
        "cv": float(std_val / mean_val) if mean_val != 0 else np.nan,
        "p10": float(s.quantile(0.10)),
        "p05": float(s.quantile(0.05)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE)
    panel = clean_percentage_columns(panel)
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])

    required = [ASSET_COL, DATE_COL, TARGET_COL, PRICE_COL]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required panel columns: {missing}")

    panel["owned_capacity_mw"] = find_owned_capacity(panel)

    keep_cols = [
        ASSET_COL,
        DATE_COL,
        TARGET_COL,
        PRICE_COL,
        "owned_capacity_mw",
        "capacity_mw",
        "ownership_fraction",
        "ownership_pct",
        TECH_COL,
        "country",
    ]
    keep_cols = [c for c in keep_cols if c in panel.columns]

    return panel[keep_cols].drop_duplicates([ASSET_COL, DATE_COL]).copy()


def standardise_actual_generation_column(merged: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    merged = merged.copy()

    if TARGET_COL in merged.columns:
        merged["actual_generation_per_mw"] = merged[TARGET_COL]
    elif f"{TARGET_COL}_panel" in merged.columns:
        merged["actual_generation_per_mw"] = merged[f"{TARGET_COL}_panel"]
    elif "actual" in merged.columns:
        merged["actual_generation_per_mw"] = merged["actual"]
    elif f"{TARGET_COL}_predfile" in merged.columns:
        merged["actual_generation_per_mw"] = merged[f"{TARGET_COL}_predfile"]
    else:
        raise ValueError(
            f"Could not find actual generation column after merge in {scenario_name}. "
            f"Available columns: {list(merged.columns)}"
        )

    return merged


def add_revenue_columns_for_fixed_share(
    df: pd.DataFrame,
    fixed_share: float,
    price_col: str = PRICE_COL,
) -> pd.DataFrame:
    df = df.copy()

    price_by_date = (
        df[[DATE_COL, price_col]]
        .drop_duplicates(DATE_COL)
        .sort_values(DATE_COL)
        .copy()
    )

    price_by_date["fixed_reference_price_gbp_mwh"] = (
        price_by_date[price_col].rolling(12, min_periods=3).mean()
    )
    price_by_date["fixed_reference_price_gbp_mwh"] = (
        price_by_date["fixed_reference_price_gbp_mwh"]
        .fillna(price_by_date[price_col].expanding().mean())
    )

    df = df.drop(columns=["fixed_reference_price_gbp_mwh"], errors="ignore").merge(
        price_by_date[[DATE_COL, "fixed_reference_price_gbp_mwh"]],
        on=DATE_COL,
        how="left",
        validate="many_to_one",
    )

    # Merchant revenue
    df["actual_revenue_gbp_merchant"] = (
        df["actual_generation_owned_mwh_proxy"] * df[price_col]
    )
    df["predicted_revenue_gbp_merchant"] = (
        df["predicted_generation_owned_mwh_proxy"] * df[price_col]
    )

    # Fixed-price revenue proxy
    df["actual_revenue_gbp_fixed"] = (
        df["actual_generation_owned_mwh_proxy"] * df["fixed_reference_price_gbp_mwh"]
    )
    df["predicted_revenue_gbp_fixed"] = (
        df["predicted_generation_owned_mwh_proxy"] * df["fixed_reference_price_gbp_mwh"]
    )

    # Blended revenue
    df["actual_revenue_gbp_blended"] = (
        fixed_share * df["actual_revenue_gbp_fixed"]
        + (1 - fixed_share) * df["actual_revenue_gbp_merchant"]
    )
    df["predicted_revenue_gbp_blended"] = (
        fixed_share * df["predicted_revenue_gbp_fixed"]
        + (1 - fixed_share) * df["predicted_revenue_gbp_merchant"]
    )

    df["revenue_error_gbp_blended"] = (
        df["predicted_revenue_gbp_blended"] - df["actual_revenue_gbp_blended"]
    )
    df["abs_revenue_error_gbp_blended"] = df["revenue_error_gbp_blended"].abs()
    df["fixed_share_assumption"] = fixed_share

    return df


def run_one_scenario(
    scenario_name: str,
    scenario_dir: Path,
    final_models: list[str],
    panel_small: pd.DataFrame,
):
    preds = load_prediction_tables(scenario_dir)

    if not INCLUDE_VALIDATION:
        preds = preds[preds["stage"] == "test"].copy()

    preds = preds[preds["model"].isin(final_models)].copy()
    if preds.empty:
        raise ValueError(f"No predictions left after filtering models for scenario {scenario_name}")

    merged = preds.merge(
        panel_small,
        on=[ASSET_COL, DATE_COL],
        how="left",
        validate="many_to_one",
        suffixes=("_predfile", "_panel"),
    )

    if merged["owned_capacity_mw"].isna().any() or merged[PRICE_COL].isna().any():
        miss = merged[["owned_capacity_mw", PRICE_COL]].isna().sum().to_dict()
        raise ValueError(f"Missing revenue inputs after merge in {scenario_name}: {miss}")

    required_after_merge = ["prediction", "owned_capacity_mw", PRICE_COL]
    missing_after_merge = [c for c in required_after_merge if c not in merged.columns]
    if missing_after_merge:
        raise ValueError(
            f"Missing required columns after merge in {scenario_name}: {missing_after_merge}. "
            f"Available columns: {list(merged.columns)}"
        )

    merged = standardise_actual_generation_column(merged, scenario_name)

    # Generation proxy at owned-capacity level
    merged["actual_generation_owned_mwh_proxy"] = (
        merged["actual_generation_per_mw"] * merged["owned_capacity_mw"]
    )
    merged["predicted_generation_owned_mwh_proxy"] = (
        merged["prediction"] * merged["owned_capacity_mw"]
    )
    merged["scenario"] = scenario_name

    # Create one dataset per fixed-share assumption
    case_tables = []
    for fixed_share in FIXED_SHARE_CASES:
        case_df = add_revenue_columns_for_fixed_share(
            merged,
            fixed_share=fixed_share,
            price_col=PRICE_COL,
        )
        case_tables.append(case_df)

    merged_all_cases = pd.concat(case_tables, ignore_index=True)

    # Asset-level financial metrics
    asset_rows = []
    for (fixed_share, model, stage), grp in merged_all_cases.groupby(
        ["fixed_share_assumption", "model", "stage"]
    ):
        asset_rows.append(
            {
                "scenario": scenario_name,
                "fixed_share_assumption": fixed_share,
                "model": model,
                "stage": stage,
                "level": "asset_month",
                **evaluate_series(
                    grp["actual_revenue_gbp_blended"],
                    grp["predicted_revenue_gbp_blended"],
                ),
            }
        )
    asset_metrics = pd.DataFrame(asset_rows)

    # Portfolio monthly aggregation
    portfolio_monthly = (
        merged_all_cases.groupby(
            ["scenario", "fixed_share_assumption", "model", "stage", DATE_COL],
            as_index=False,
        )[["actual_revenue_gbp_blended", "predicted_revenue_gbp_blended"]]
        .sum()
        .sort_values(["fixed_share_assumption", "model", "stage", DATE_COL])
    )

    portfolio_monthly["revenue_error_gbp"] = (
        portfolio_monthly["predicted_revenue_gbp_blended"]
        - portfolio_monthly["actual_revenue_gbp_blended"]
    )
    portfolio_monthly["abs_revenue_error_gbp"] = portfolio_monthly["revenue_error_gbp"].abs()

    portfolio_rows = []
    risk_rows = []

    for (fixed_share, model, stage), grp in portfolio_monthly.groupby(
        ["fixed_share_assumption", "model", "stage"]
    ):
        metrics = evaluate_series(
            grp["actual_revenue_gbp_blended"],
            grp["predicted_revenue_gbp_blended"],
        )

        threshold = grp["actual_revenue_gbp_blended"].quantile(0.10)
        downside = grp[grp["actual_revenue_gbp_blended"] <= threshold]

        if len(downside) > 0:
            downside_metrics = evaluate_series(
                downside["actual_revenue_gbp_blended"],
                downside["predicted_revenue_gbp_blended"],
            )
        else:
            downside_metrics = {"MAE": np.nan, "RMSE": np.nan, "Mean_Bias": np.nan}

        portfolio_rows.append(
            {
                "scenario": scenario_name,
                "fixed_share_assumption": fixed_share,
                "model": model,
                "stage": stage,
                "level": "portfolio_month",
                **metrics,
                "downside_month_count": int(len(downside)),
                "downside_MAE": downside_metrics["MAE"],
                "downside_RMSE": downside_metrics["RMSE"],
                "downside_Mean_Bias": downside_metrics["Mean_Bias"],
            }
        )

        actual_r = risk_summary(grp["actual_revenue_gbp_blended"])
        pred_r = risk_summary(grp["predicted_revenue_gbp_blended"])

        risk_rows.append(
            {
                "scenario": scenario_name,
                "fixed_share_assumption": fixed_share,
                "model": model,
                "stage": stage,
                "series": "actual",
                **actual_r,
            }
        )
        risk_rows.append(
            {
                "scenario": scenario_name,
                "fixed_share_assumption": fixed_share,
                "model": model,
                "stage": stage,
                "series": "predicted",
                **pred_r,
            }
        )
        risk_rows.append(
            {
                "scenario": scenario_name,
                "fixed_share_assumption": fixed_share,
                "model": model,
                "stage": stage,
                "series": "gap_pred_minus_actual",
                "mean": pred_r["mean"] - actual_r["mean"],
                "std_dev": pred_r["std_dev"] - actual_r["std_dev"],
                "cv": pred_r["cv"] - actual_r["cv"]
                if pd.notna(pred_r["cv"]) and pd.notna(actual_r["cv"])
                else np.nan,
                "p10": pred_r["p10"] - actual_r["p10"],
                "p05": pred_r["p05"] - actual_r["p05"],
                "min": pred_r["min"] - actual_r["min"],
                "max": pred_r["max"] - actual_r["max"],
            }
        )

    return (
        merged_all_cases,
        asset_metrics,
        portfolio_monthly,
        pd.DataFrame(portfolio_rows),
        pd.DataFrame(risk_rows),
    )


def main():
    panel_small = prepare_panel()

    all_asset_preds = []
    all_asset_metrics = []
    all_portfolio_monthly = []
    all_portfolio_metrics = []
    all_risk_metrics = []
    status_rows = []

    for scenario_name, cfg in SCENARIOS.items():
        try:
            merged, asset_metrics, portfolio_monthly, portfolio_metrics, risk_metrics = run_one_scenario(
                scenario_name=scenario_name,
                scenario_dir=cfg["dir"],
                final_models=cfg["models"],
                panel_small=panel_small,
            )
            all_asset_preds.append(merged)
            all_asset_metrics.append(asset_metrics)
            all_portfolio_monthly.append(portfolio_monthly)
            all_portfolio_metrics.append(portfolio_metrics)
            all_risk_metrics.append(risk_metrics)
            status_rows.append(
                {"scenario": scenario_name, "status": "ok", "rows": len(merged)}
            )
        except Exception as e:
            status_rows.append(
                {"scenario": scenario_name, "status": "failed", "reason": str(e)}
            )

    pd.DataFrame(status_rows).to_csv(OUTPUT_DIR / "financial_scenario_status.csv", index=False)

    if not all_asset_preds:
        raise ValueError("All financial scenarios failed.")

    asset_preds_df = pd.concat(all_asset_preds, ignore_index=True)
    asset_metrics_df = pd.concat(all_asset_metrics, ignore_index=True)
    portfolio_monthly_df = pd.concat(all_portfolio_monthly, ignore_index=True)
    portfolio_metrics_df = pd.concat(all_portfolio_metrics, ignore_index=True)
    risk_df = pd.concat(all_risk_metrics, ignore_index=True)

    asset_preds_df.to_csv(OUTPUT_DIR / "asset_level_revenue_predictions.csv", index=False)
    asset_metrics_df.to_csv(OUTPUT_DIR / "asset_level_financial_metrics.csv", index=False)
    portfolio_monthly_df.to_csv(OUTPUT_DIR / "portfolio_monthly_revenue.csv", index=False)
    portfolio_metrics_df.to_csv(OUTPUT_DIR / "portfolio_financial_metrics.csv", index=False)
    risk_df.to_csv(OUTPUT_DIR / "portfolio_risk_metrics.csv", index=False)

    if "test" in portfolio_metrics_df["stage"].unique():
        summary_stage = "test"
    else:
        summary_stage = portfolio_metrics_df["stage"].iloc[0]

    summary = (
        portfolio_metrics_df[portfolio_metrics_df["stage"] == summary_stage]
        .merge(
            risk_df[
                (risk_df["stage"] == summary_stage) & (risk_df["series"] == "predicted")
            ][
                [
                    "scenario",
                    "fixed_share_assumption",
                    "model",
                    "stage",
                    "std_dev",
                    "p10",
                    "p05",
                    "min",
                ]
            ].rename(
                columns={
                    "std_dev": "predicted_revenue_volatility",
                    "p10": "predicted_revenue_p10",
                    "p05": "predicted_revenue_p05",
                    "min": "predicted_revenue_worst_month",
                }
            ),
            on=["scenario", "fixed_share_assumption", "model", "stage"],
            how="left",
        )
        .sort_values(["scenario", "fixed_share_assumption", "RMSE", "MAE"])
        .reset_index(drop=True)
    )
    summary.to_csv(OUTPUT_DIR / "financial_model_comparison_summary.csv", index=False)

    with open(OUTPUT_DIR / "financial_run_metadata.json", "w") as f:
        json.dump(
            {
                "panel_file": str(PANEL_FILE),
                "scenarios": {
                    k: {"dir": str(v["dir"]), "models": v["models"]}
                    for k, v in SCENARIOS.items()
                },
                "include_validation": INCLUDE_VALIDATION,
                "fixed_share_cases": FIXED_SHARE_CASES,
                "asset_revenue_rows": int(len(asset_preds_df)),
                "portfolio_month_rows": int(len(portfolio_monthly_df)),
            },
            f,
            indent=2,
        )

    print(f"Done. Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()