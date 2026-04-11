from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(".")
INPUT_FILE = BASE_DIR / "model_panel_clean.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs_refined"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "generation_per_mw"
DATE_COL = "date"
ASSET_COL = "asset_id"
TECH_COL = "technology"

MAIN_SPLIT = {
    "name": "main_2018_2022_train__2023_valid__2024_test",
    "train_end": "2022-12-01",
    "valid_end": "2023-12-01",
}
ALT_SPLIT = {
    "name": "alt_2018_2021_train__2022_valid__2023_2024_test",
    "train_end": "2021-12-01",
    "valid_end": "2022-12-01",
}

RUN_MLP = True
RUN_XGBOOST = True
RANDOM_STATE = 42
MIN_SUBSET_ROWS = 24

# Features excluded from the forecasting stage because the target is already normalised by MW.
# They remain valid for downstream financial conversion, but not for predicting generation_per_mw.
EXCLUDED_FORECAST_NUMERIC_FEATURES = {
    "capacity_mw",
    "owned_capacity_mw",
    "ownership_fraction",
    "ownership_pct",
    "portfolio_capacity_pct",
}
EXCLUDED_FORECAST_CATEGORICAL_FEATURES = set()



def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def directional_correlation(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "sMAPE_pct": smape(y_true, y_pred),
        "Mean_Bias": mean_bias(y_true, y_pred),
        "Corr": directional_correlation(y_true, y_pred),
    }


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
                    .astype(float) / 100.0
                )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["year_num"] = df[DATE_COL].dt.year
    df["month_num"] = df[DATE_COL].dt.month
    df["quarter_num"] = df[DATE_COL].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["winter_flag"] = df["month_num"].isin([12, 1, 2]).astype(int)
    df["summer_flag"] = df["month_num"].isin([6, 7, 8]).astype(int)
    df["time_index"] = ((df[DATE_COL].dt.year - df[DATE_COL].dt.year.min()) * 12 + df[DATE_COL].dt.month)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values([ASSET_COL, DATE_COL]).reset_index(drop=True)
    climate_vars = ["temp_2m_c", "precip_mm", "solar_radiation_j_m2", "wind_speed_10m"]

    for var in climate_vars:
        if var in df.columns:
            grp = df.groupby(ASSET_COL)[var]
            for lag in [3, 6, 12]:
                df[f"{var}_lag{lag}"] = grp.shift(lag)
            df[f"{var}_roll6_mean"] = grp.transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
            df[f"{var}_roll12_mean"] = grp.transform(lambda s: s.shift(1).rolling(12, min_periods=4).mean())
            df[f"{var}_roll6_std"] = grp.transform(lambda s: s.shift(1).rolling(6, min_periods=3).std())
            if f"{var}_anomaly" in df.columns:
                df[f"{var}_anomaly_abs"] = df[f"{var}_anomaly"].abs()

    grp_y = df.groupby(ASSET_COL)[TARGET_COL]
    for lag in [1, 3, 6, 12]:
        df[f"{TARGET_COL}_lag{lag}"] = grp_y.shift(lag)
    df[f"{TARGET_COL}_roll3_mean"] = grp_y.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    df[f"{TARGET_COL}_roll6_mean"] = grp_y.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())

    if "capacity_mw" in df.columns and "ownership_fraction" in df.columns:
        df["owned_capacity_mw"] = df["capacity_mw"] * df["ownership_fraction"]

    if TECH_COL in df.columns:
        tech_dummies = pd.get_dummies(df[TECH_COL], prefix="tech")
        for base in ["wind_speed_10m", "solar_radiation_j_m2", "temp_2m_c", "precip_mm"]:
            if base in df.columns:
                for tcol in tech_dummies.columns:
                    df[f"{base}__x__{tcol}"] = df[base] * tech_dummies[tcol].astype(float)

    if "solar_radiation_j_m2" in df.columns:
        df["log_solar_radiation"] = np.log1p(np.clip(df["solar_radiation_j_m2"], a_min=0, a_max=None))
    if "wind_speed_10m" in df.columns:
        df["wind_speed_sq"] = df["wind_speed_10m"] ** 2
    if "temp_2m_c" in df.columns:
        df["temp_sq"] = df["temp_2m_c"] ** 2

    return df


def split_by_dates(df: pd.DataFrame, train_end: str, valid_end: str):
    train_end = pd.Timestamp(train_end)
    valid_end = pd.Timestamp(valid_end)
    train = df[df[DATE_COL] <= train_end].copy()
    valid = df[(df[DATE_COL] > train_end) & (df[DATE_COL] <= valid_end)].copy()
    test = df[df[DATE_COL] > valid_end].copy()
    return train, valid, test


def baseline_asset_mean(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    means = train_df.groupby(ASSET_COL)[TARGET_COL].mean()
    pred = target_df[ASSET_COL].map(means)
    return pred.fillna(train_df[TARGET_COL].mean())


def baseline_asset_month_seasonal(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    means = train_df.groupby([ASSET_COL, "month_num"])[TARGET_COL].mean()
    idx = pd.MultiIndex.from_frame(target_df[[ASSET_COL, "month_num"]])
    pred = pd.Series(idx.map(means), index=target_df.index)
    return pred.fillna(baseline_asset_mean(train_df, target_df))


def baseline_last_value(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    last_vals = train_df.sort_values([ASSET_COL, DATE_COL]).groupby(ASSET_COL)[TARGET_COL].last()
    pred = target_df[ASSET_COL].map(last_vals)
    return pred.fillna(baseline_asset_mean(train_df, target_df))


def get_feature_sets(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    exclude_base = {
        TARGET_COL, DATE_COL, ASSET_COL, "asset_name", "generation_mwh_est", "generation_mwh_est_owned",
        "generation_mwh_per_mw", "revenue_proxy_gbp", "power_price_gbp_mwh", "power_price_gbp_mwh_lag1",
        "selected_grid_lat", "selected_grid_lon", "start_of_operations", "pre_operational_flag", "temp_2m_k",
        "precip_m", "wind_u10", "wind_v10", "year", "month"
    }
    all_numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in exclude_base
        and c not in EXCLUDED_FORECAST_NUMERIC_FEATURES
    ]
    all_categorical = [
        c for c in df.columns
        if df[c].dtype == "object"
        and c not in exclude_base
        and c not in EXCLUDED_FORECAST_CATEGORICAL_FEATURES
    ]

    core_numeric = [
        c for c in [
            "temp_2m_c", "precip_mm", "solar_radiation_j_m2", "wind_speed_10m",
            "temp_2m_c_lag1", "precip_mm_lag1", "solar_radiation_j_m2_lag1", "wind_speed_10m_lag1",
            "temp_2m_c_roll3_mean", "precip_mm_roll3_mean", "solar_radiation_j_m2_roll3_mean", "wind_speed_10m_roll3_mean",
            "temp_2m_c_anomaly", "precip_mm_anomaly", "solar_radiation_j_m2_anomaly", "wind_speed_10m_anomaly",
            "remaining_asset_life_years", "commissioning_year",
            "latitude", "longitude", "month_sin", "month_cos", "winter_flag", "summer_flag", "time_index"
        ] if c in all_numeric
    ]
    core_categorical = [c for c in [TECH_COL, "country", "operational_status"] if c in all_categorical]

    extended_numeric = [
        c for c in all_numeric
        if (c in core_numeric) or any(tag in c for tag in [
            "_lag3", "_lag6", "_lag12", "_roll6_", "_roll12_", "_anomaly_abs", "__x__tech_",
            "log_solar_radiation", "wind_speed_sq", "temp_sq"
        ])
    ]
    autoreg_numeric = extended_numeric + [
        c for c in all_numeric if c.startswith(f"{TARGET_COL}_lag") or c.startswith(f"{TARGET_COL}_roll")
    ]

    def dedupe(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "core": {"numeric": dedupe(core_numeric), "categorical": dedupe(core_categorical)},
        "extended": {"numeric": dedupe(extended_numeric), "categorical": dedupe(core_categorical)},
        "autoreg": {"numeric": dedupe(autoreg_numeric), "categorical": dedupe(core_categorical)},
    }


@dataclass
class ModelSpec:
    name: str
    estimator_factory: Callable[[], object]
    param_grid: list[dict]
    scaled_numeric: bool


def maybe_get_xgboost_spec() -> list[ModelSpec]:
    if not RUN_XGBOOST:
        return []
    try:
        from xgboost import XGBRegressor
    except Exception:
        return []
    return [
        ModelSpec(
            name="XGBoost",
            estimator_factory=lambda: XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=list(ParameterGrid({
                "n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.03, 0.08],
                "subsample": [0.8], "colsample_bytree": [0.8], "reg_alpha": [0.0, 0.1], "reg_lambda": [1.0, 3.0]
            })),
            scaled_numeric=False,
        )
    ]


def get_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec("Ridge", lambda: Ridge(random_state=RANDOM_STATE), list(ParameterGrid({"alpha": [0.1, 1.0, 5.0, 10.0]})), True),
        ModelSpec("ElasticNet", lambda: ElasticNet(random_state=RANDOM_STATE, max_iter=5000), list(ParameterGrid({"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]})), True),
        ModelSpec("RandomForest", lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), list(ParameterGrid({"n_estimators": [300, 600], "max_depth": [8, 12, None], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.8]})), False),
        ModelSpec("ExtraTrees", lambda: ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1), list(ParameterGrid({"n_estimators": [300, 600], "max_depth": [8, 12, None], "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.8]})), False),
        ModelSpec("HistGradientBoosting", lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE), list(ParameterGrid({"max_depth": [3, 6], "learning_rate": [0.03, 0.08], "max_iter": [200, 400], "min_samples_leaf": [10, 20]})), False),
    ]
    if RUN_MLP:
        specs.append(ModelSpec("MLP", lambda: MLPRegressor(random_state=RANDOM_STATE, early_stopping=True), list(ParameterGrid({"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.0005, 0.005], "learning_rate_init": [0.001, 0.01], "max_iter": [500]})), True))
    specs.extend(maybe_get_xgboost_spec())
    return specs


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scaled_numeric: bool):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(num_steps)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def fit_pipeline(estimator, preprocessor, X_train, y_train) -> Pipeline:
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    return pipe


def time_series_cv_score(df_train_valid: pd.DataFrame, numeric_features, categorical_features, model_spec: ModelSpec, params: dict, n_splits: int = 4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dates_sorted = np.array(sorted(df_train_valid[DATE_COL].unique()))
    if len(dates_sorted) < n_splits + 2:
        return np.nan, np.nan
    fold_scores = []
    for train_idx, valid_idx in tscv.split(dates_sorted):
        train_dates = set(dates_sorted[train_idx])
        valid_dates = set(dates_sorted[valid_idx])
        fold_train = df_train_valid[df_train_valid[DATE_COL].isin(train_dates)]
        fold_valid = df_train_valid[df_train_valid[DATE_COL].isin(valid_dates)]
        if fold_train.empty or fold_valid.empty:
            continue
        estimator = model_spec.estimator_factory()
        estimator.set_params(**params)
        preprocessor = make_preprocessor(numeric_features, categorical_features, model_spec.scaled_numeric)
        pipe = fit_pipeline(estimator, preprocessor, fold_train[numeric_features + categorical_features], fold_train[TARGET_COL])
        pred = pipe.predict(fold_valid[numeric_features + categorical_features])
        metrics = evaluate_predictions(fold_valid[TARGET_COL], pred)
        fold_scores.append(metrics["RMSE"])
    if not fold_scores:
        return np.nan, np.nan
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def select_best_params(train_df: pd.DataFrame, valid_df: pd.DataFrame, numeric_features, categorical_features, model_spec: ModelSpec):
    rows = []
    best_pipe, best_params, best_valid_rmse = None, None, np.inf
    for params in model_spec.param_grid:
        estimator = model_spec.estimator_factory()
        estimator.set_params(**params)
        preprocessor = make_preprocessor(numeric_features, categorical_features, model_spec.scaled_numeric)
        pipe = fit_pipeline(estimator, preprocessor, train_df[numeric_features + categorical_features], train_df[TARGET_COL])
        valid_pred = pipe.predict(valid_df[numeric_features + categorical_features])
        valid_metrics = evaluate_predictions(valid_df[TARGET_COL], valid_pred)
        cv_mean_rmse, cv_std_rmse = time_series_cv_score(pd.concat([train_df, valid_df], ignore_index=True), numeric_features, categorical_features, model_spec, params, n_splits=4)
        row = {"model": model_spec.name, **params, **{f"valid_{k}": v for k, v in valid_metrics.items()}, "cv_mean_RMSE": cv_mean_rmse, "cv_std_RMSE": cv_std_rmse}
        rows.append(row)
        if valid_metrics["RMSE"] < best_valid_rmse:
            best_valid_rmse, best_pipe, best_params = valid_metrics["RMSE"], pipe, params
    tuning_df = pd.DataFrame(rows).sort_values("valid_RMSE", ascending=True).reset_index(drop=True)
    return best_pipe, best_params, tuning_df


def refit_and_predict(train_valid_df: pd.DataFrame, test_df: pd.DataFrame, numeric_features, categorical_features, model_spec: ModelSpec, params: dict):
    estimator = model_spec.estimator_factory()
    estimator.set_params(**params)
    preprocessor = make_preprocessor(numeric_features, categorical_features, model_spec.scaled_numeric)
    pipe = fit_pipeline(estimator, preprocessor, train_valid_df[numeric_features + categorical_features], train_valid_df[TARGET_COL])
    preds = pipe.predict(test_df[numeric_features + categorical_features])
    metrics = evaluate_predictions(test_df[TARGET_COL], preds)
    return pipe, preds, metrics


def try_export_importance(pipe: Pipeline) -> pd.DataFrame | None:
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        return None
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        imp = pd.DataFrame({"feature": feature_names, "coefficient": coef, "abs_coefficient": np.abs(coef)})
        return imp.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return None


def run_scenario(df: pd.DataFrame, scenario_name: str, split_config: dict, feature_spec_name: str, subset_label: str, subset_filter: Callable[[pd.DataFrame], pd.DataFrame] | None):
    out_dir = OUTPUT_DIR / scenario_name / feature_spec_name / subset_label
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_df = df.copy()
    if subset_filter is not None:
        scenario_df = subset_filter(scenario_df).copy()
    scenario_df = scenario_df.sort_values([DATE_COL, ASSET_COL]).reset_index(drop=True)
    train_df, valid_df, test_df = split_by_dates(scenario_df, split_config["train_end"], split_config["valid_end"])
    if train_df.empty or valid_df.empty or test_df.empty:
        return {"status": "skipped", "reason": "empty split"}
    if len(train_df) < MIN_SUBSET_ROWS or len(valid_df) < 6 or len(test_df) < 6:
        return {"status": "skipped", "reason": "too few rows"}

    feature_sets = get_feature_sets(scenario_df)
    fset = feature_sets[feature_spec_name]
    numeric_features, categorical_features = fset["numeric"], fset["categorical"]

    baseline_rows, baseline_pred_tables = [], []
    baseline_methods = {"Baseline_AssetMean": baseline_asset_mean, "Baseline_AssetMonthSeasonal": baseline_asset_month_seasonal, "Baseline_LastValue": baseline_last_value}
    for bname, func in baseline_methods.items():
        valid_pred = func(train_df, valid_df)
        test_pred = func(pd.concat([train_df, valid_df], ignore_index=True), test_df)
        baseline_rows.append({"scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "model": bname, "stage": "validation", **evaluate_predictions(valid_df[TARGET_COL], valid_pred)})
        baseline_rows.append({"scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "model": bname, "stage": "test", **evaluate_predictions(test_df[TARGET_COL], test_pred)})
        baseline_pred_tables.append(pd.concat([
            valid_df[[ASSET_COL, DATE_COL, TARGET_COL]].assign(stage="validation", model=bname, prediction=np.asarray(valid_pred)),
            test_df[[ASSET_COL, DATE_COL, TARGET_COL]].assign(stage="test", model=bname, prediction=np.asarray(test_pred)),
        ], ignore_index=True))
    baseline_metrics_df = pd.DataFrame(baseline_rows)
    baseline_metrics_df.to_csv(out_dir / "baseline_metrics.csv", index=False)
    pd.concat(baseline_pred_tables, ignore_index=True).to_csv(out_dir / "baseline_predictions.csv", index=False)

    model_metrics_rows, prediction_tables, tuning_tables = [], [], []
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    best_pipes_for_ensemble = {}

    for model_spec in get_model_specs():
        best_valid_pipe, best_params, tuning_df = select_best_params(train_df, valid_df, numeric_features, categorical_features, model_spec)
        tuning_df.insert(0, "scenario", scenario_name)
        tuning_df.insert(1, "feature_spec", feature_spec_name)
        tuning_df.insert(2, "subset", subset_label)
        tuning_tables.append(tuning_df)

        valid_pred = best_valid_pipe.predict(valid_df[numeric_features + categorical_features])
        model_metrics_rows.append({"scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "model": model_spec.name, "stage": "validation", **evaluate_predictions(valid_df[TARGET_COL], valid_pred), **{f"best_{k}": v for k, v in best_params.items()}})

        refit_pipe, test_pred, test_metrics = refit_and_predict(train_valid_df, test_df, numeric_features, categorical_features, model_spec, best_params)
        model_metrics_rows.append({"scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "model": model_spec.name, "stage": "test", **test_metrics, **{f"best_{k}": v for k, v in best_params.items()}})

        prediction_tables.append(pd.concat([
            valid_df[[ASSET_COL, DATE_COL, TARGET_COL]].assign(stage="validation", model=model_spec.name, prediction=np.asarray(valid_pred)),
            test_df[[ASSET_COL, DATE_COL, TARGET_COL]].assign(stage="test", model=model_spec.name, prediction=np.asarray(test_pred)),
        ], ignore_index=True))

        importance_df = try_export_importance(refit_pipe)
        if importance_df is not None:
            importance_df.to_csv(out_dir / f"importance_{model_spec.name}.csv", index=False)

        best_pipes_for_ensemble[model_spec.name] = {"pipe_valid": best_valid_pipe, "pipe_test": refit_pipe}

    model_metrics_df = pd.DataFrame(model_metrics_rows)
    model_metrics_df.to_csv(out_dir / "model_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(out_dir / "model_predictions.csv", index=False)
    pd.concat(tuning_tables, ignore_index=True).to_csv(out_dir / "tuning_results.csv", index=False)

    valid_only = model_metrics_df[model_metrics_df["stage"] == "validation"].sort_values("RMSE")
    top_models = valid_only["model"].head(3).tolist()
    if len(top_models) >= 2:
        ensemble_valid_pred = np.mean(np.vstack([best_pipes_for_ensemble[m]["pipe_valid"].predict(valid_df[numeric_features + categorical_features]) for m in top_models]), axis=0)
        ensemble_test_pred = np.mean(np.vstack([best_pipes_for_ensemble[m]["pipe_test"].predict(test_df[numeric_features + categorical_features]) for m in top_models]), axis=0)
        ensemble_metrics_df = pd.DataFrame([
            {"scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "model": "Ensemble_AvgTop3", "stage": "validation", **evaluate_predictions(valid_df[TARGET_COL], ensemble_valid_pred)},
            {"scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "model": "Ensemble_AvgTop3", "stage": "test", **evaluate_predictions(test_df[TARGET_COL], ensemble_test_pred)},
        ])
        ensemble_metrics_df.to_csv(out_dir / "ensemble_metrics.csv", index=False)
        pd.concat([model_metrics_df, baseline_metrics_df, ensemble_metrics_df], ignore_index=True).to_csv(out_dir / "all_metrics_combined.csv", index=False)
        pd.concat([
            valid_df[[ASSET_COL, DATE_COL, TARGET_COL]].assign(stage="validation", model="Ensemble_AvgTop3", prediction=ensemble_valid_pred),
            test_df[[ASSET_COL, DATE_COL, TARGET_COL]].assign(stage="test", model="Ensemble_AvgTop3", prediction=ensemble_test_pred),
        ], ignore_index=True).to_csv(out_dir / "ensemble_predictions.csv", index=False)
    else:
        pd.concat([model_metrics_df, baseline_metrics_df], ignore_index=True).to_csv(out_dir / "all_metrics_combined.csv", index=False)

    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump({
            "scenario": scenario_name, "feature_spec": feature_spec_name, "subset": subset_label, "split_name": split_config["name"],
            "train_rows": int(len(train_df)), "valid_rows": int(len(valid_df)), "test_rows": int(len(test_df)),
            "train_assets": int(train_df[ASSET_COL].nunique()), "valid_assets": int(valid_df[ASSET_COL].nunique()), "test_assets": int(test_df[ASSET_COL].nunique()),
            "numeric_features_count": len(numeric_features), "categorical_features_count": len(categorical_features),
            "numeric_features": numeric_features, "categorical_features": categorical_features,
            "excluded_forecast_numeric_features": sorted(EXCLUDED_FORECAST_NUMERIC_FEATURES),
            "excluded_forecast_categorical_features": sorted(EXCLUDED_FORECAST_CATEGORICAL_FEATURES),
        }, f, indent=2)
    return {"status": "ok", "out_dir": str(out_dir)}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")
    df = pd.read_csv(INPUT_FILE)
    df = clean_percentage_columns(df)
    df = add_calendar_features(df)
    df = add_engineered_features(df)
    if "pre_operational_flag" in df.columns:
        df = df[df["pre_operational_flag"] == 0].copy()
    df = df.sort_values([DATE_COL, ASSET_COL]).reset_index(drop=True)

    with open(OUTPUT_DIR / "dataset_audit.json", "w") as f:
        json.dump({
            "rows": int(len(df)), "assets": int(df[ASSET_COL].nunique()),
            "date_min": str(pd.to_datetime(df[DATE_COL]).min().date()), "date_max": str(pd.to_datetime(df[DATE_COL]).max().date()),
            "duplicate_asset_date_rows": int(df.duplicated([ASSET_COL, DATE_COL]).sum()), "missing_target": int(df[TARGET_COL].isna().sum()),
            "technologies": sorted(df[TECH_COL].dropna().astype(str).unique().tolist()) if TECH_COL in df.columns else [],
            "excluded_forecast_numeric_features": sorted(EXCLUDED_FORECAST_NUMERIC_FEATURES),
            "excluded_forecast_categorical_features": sorted(EXCLUDED_FORECAST_CATEGORICAL_FEATURES),
        }, f, indent=2)

    scenarios = [
        ("main_all_assets", MAIN_SPLIT, "extended", "all_assets", None),
        ("main_core_features", MAIN_SPLIT, "core", "all_assets", None),
        ("main_autoreg_features", MAIN_SPLIT, "autoreg", "all_assets", None),
        ("alt_split_extended", ALT_SPLIT, "extended", "all_assets", None),
    ]
    if TECH_COL in df.columns:
        for tech in sorted(df[TECH_COL].dropna().astype(str).unique()):
            tech_name = str(tech)
            scenarios.append((f"main_extended_{tech_name}", MAIN_SPLIT, "extended", f"technology_{tech_name}", lambda x, tech_name=tech_name: x[x[TECH_COL].astype(str) == tech_name]))

    status_rows = []
    for scenario_name, split_cfg, feature_spec, subset_label, subset_filter in scenarios:
        result = run_scenario(df, scenario_name, split_cfg, feature_spec, subset_label, subset_filter)
        status_rows.append({"scenario": scenario_name, "feature_spec": feature_spec, "subset": subset_label, **result})
    pd.DataFrame(status_rows).to_csv(OUTPUT_DIR / "scenario_status.csv", index=False)

    all_metrics_paths = list(OUTPUT_DIR.glob("**/all_metrics_combined.csv"))
    if all_metrics_paths:
        master_metrics = pd.concat([pd.read_csv(p) for p in all_metrics_paths], ignore_index=True)
        master_metrics.to_csv(OUTPUT_DIR / "MASTER_all_metrics.csv", index=False)
        master_metrics[master_metrics["stage"] == "test"].sort_values(["scenario", "RMSE", "MAE"]).reset_index(drop=True).to_csv(OUTPUT_DIR / "MASTER_test_rankings.csv", index=False)
        master_metrics[master_metrics["stage"] == "validation"].sort_values(["scenario", "RMSE", "MAE"]).reset_index(drop=True).to_csv(OUTPUT_DIR / "MASTER_validation_rankings.csv", index=False)

    print(f"Done. Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
