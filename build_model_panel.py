from __future__ import annotations

import calendar
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

# All paths resolve relative to the working directory. Each stage of the
# pipeline writes its output before the next stage reads it, so the run order
# defined in main() is required for a clean first-time build.
BASE_DIR = Path(".")

ASSETS_FILE = BASE_DIR / "assets.csv"

ERA5_LAND_FILE = BASE_DIR / "climate_data.nc"
ERA5_FILES = [BASE_DIR / "ERA5-1.nc", BASE_DIR / "ERA5-2.nc"]

CLIMATE_OUTPUT_FILE = BASE_DIR / "climate_asset_monthly.csv"
GENERATION_OUTPUT_FILE = BASE_DIR / "generation_monthly.csv"
PRICES_OUTPUT_FILE = BASE_DIR / "prices_monthly.csv"
MODEL_PANEL_OUTPUT_FILE = BASE_DIR / "model_panel.csv"
MODEL_PANEL_CLEAN_OUTPUT_FILE = BASE_DIR / "model_panel_clean.csv"

GENERATION_PROGRESS_FILE = BASE_DIR / "progress_generation.csv"
SUMMARY_FILE = BASE_DIR / "data_build_summary.json"

# The empirical window is fixed to January 2018 through December 2024 in order
# to align all sources to a common monthly time horizon.
START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2024-12-01")
START_YEAR = 2018
END_YEAR = 2024

# SECONDS_BETWEEN_REQUESTS is set conservatively to stay within the
# Renewables.ninja rate limit. TEST_ASSET_LIMIT restricts the asset loop
# during development without altering production behaviour.
TOKEN = "YOUR_RENEWABLES_NINJA_TOKEN"
API_BASE = "https://www.renewables.ninja/api/"
SECONDS_BETWEEN_REQUESTS = 20
TEST_ASSET_LIMIT = None

BMRS_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
BMRS_START = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
BMRS_END = pd.Timestamp("2024-12-31 23:59:00", tz="UTC")

SEARCH_RADIUS_STEPS = 4
ONSHORE_KEYWORDS = ["solar", "onshore"]
OFFSHORE_KEYWORDS = ["offshore"]

# Setting a stage flag to False skips that stage and assumes its output CSV
# already exists on disk, which avoids re-running the API-heavy steps.
RUN_CLIMATE = True
RUN_GENERATION = True
RUN_PRICES = True
RUN_MERGE = True


def clean_percentage(value):
    """
    Convert ownership strings such as "75%" into decimal fractions.

    Ownership data is read from the manually curated asset file and may arrive as
    mixed-format strings. This helper standardises the representation while
    preserving missing values.
    """
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().replace("%", "")
    if s == "":
        return pd.NA
    try:
        return float(s) / 100.0
    except ValueError:
        return pd.NA


def dataset_snapshot(df: pd.DataFrame, name: str) -> dict:
    """
    Summarise the dimensional properties of an intermediate dataset.

    These snapshots are saved to the JSON audit file so that each pipeline stage
    records its row count, column count, asset coverage, and date coverage.
    """
    out = {"name": name, "rows": int(len(df)), "cols": int(df.shape[1])}
    if "asset_id" in df.columns:
        out["assets"] = int(df["asset_id"].astype(str).nunique())
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        out["date_min"] = None if dt.isna().all() else str(dt.min().date())
        out["date_max"] = None if dt.isna().all() else str(dt.max().date())
    return out


def normalise_longitude_for_dataset(lon, ds, lon_name):
    """
    Align longitude values to the coordinate convention used by a dataset.

    Some NetCDF files use longitudes in [-180, 180] while others use [0, 360].
    The conversion ensures spatial selection is applied consistently.
    """
    lon_values = ds[lon_name].values
    if np.nanmax(lon_values) > 180 and lon < 0:
        return lon % 360
    return lon


def detect_names_and_vars(ds):
    """
    Infer coordinate names and climate variable names from a NetCDF dataset.

    The climate files do not necessarily share a single naming convention, so
    the required coordinates and variables are detected programmatically.
    """
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    if "valid_time" in ds.coords:
        time_name = "valid_time"
    elif "time" in ds.coords:
        time_name = "time"
    else:
        raise ValueError("Could not detect time coordinate. Expected 'valid_time' or 'time'.")

    var_map = {"temp": None, "precip": None, "solar": None, "u10": None, "v10": None}

    for var in ds.data_vars:
        v = var.lower()
        if "2m_temperature" in v or v == "t2m":
            var_map["temp"] = var
        elif "total_precipitation" in v or v == "tp":
            var_map["precip"] = var
        elif "surface_solar_radiation_downwards" in v or v == "ssrd":
            var_map["solar"] = var
        elif "10m_u_component_of_wind" in v or v == "u10":
            var_map["u10"] = var
        elif "10m_v_component_of_wind" in v or v == "v10":
            var_map["v10"] = var

    return lat_name, lon_name, time_name, var_map


def build_keep_cols(df, time_name, var_map):
    """Return the extracted columns required for climate modelling."""
    keep_cols = [time_name]
    for key in ["temp", "precip", "solar", "u10", "v10"]:
        if var_map[key] is not None and var_map[key] in df.columns:
            keep_cols.append(var_map[key])
    return keep_cols


def all_value_cols_nan(df, time_name):
    """
    Test whether an extracted climate point contains no usable observations.
    """
    value_cols = [c for c in df.columns if c != time_name]
    if not value_cols:
        return True
    return df[value_cols].isna().all().all()


def point_to_dataframe(point, time_name, var_map):
    """Convert a selected xarray point into a tidy pandas DataFrame."""
    df = point.to_dataframe().reset_index()
    keep_cols = build_keep_cols(df, time_name, var_map)
    return df[keep_cols].copy()


def extract_nearest_point(ds, lat, lon, lat_name, lon_name):
    """Select the nearest climate grid cell to an asset coordinate."""
    return ds.sel({lat_name: lat, lon_name: lon}, method="nearest")


def extract_nearest_valid_land_point(ds, lat, lon, lat_name, lon_name, time_name, var_map, max_steps=4):
    """
    Search locally for the nearest non-empty ERA5-Land grid cell.

    This is mainly required for coastal assets where the nearest land-based cell
    may be empty because of shoreline geometry.
    """
    lon = normalise_longitude_for_dataset(lon, ds, lon_name)
    lat_values = np.asarray(ds[lat_name].values)
    lon_values = np.asarray(ds[lon_name].values)
    lat_idx = int(np.abs(lat_values - lat).argmin())
    lon_idx = int(np.abs(lon_values - lon).argmin())

    candidates = []
    for r in range(max_steps + 1):
        for i in range(max(0, lat_idx - r), min(len(lat_values), lat_idx + r + 1)):
            for j in range(max(0, lon_idx - r), min(len(lon_values), lon_idx + r + 1)):
                point = ds.isel({lat_name: i, lon_name: j})
                df = point_to_dataframe(point, time_name, var_map)
                if not all_value_cols_nan(df, time_name):
                    grid_lat = float(lat_values[i])
                    grid_lon = float(lon_values[j])
                    distance = np.sqrt((grid_lat - lat) ** 2 + (grid_lon - lon) ** 2)
                    candidates.append((distance, point, grid_lat, grid_lon))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_point, best_lat, best_lon = candidates[0]
            return best_point, best_lat, best_lon, f"fallback_valid_cell_r{r}"

    return None, None, None, "all_candidate_cells_nan"


def standardise_extracted_df(df, asset_id, source_dataset, selected_grid_lat, selected_grid_lon, extraction_note, time_name, var_map):
    """
    Standardise one extracted climate table and attach traceability metadata.
    """
    df = df.copy()
    df.rename(columns={time_name: "date"}, inplace=True)
    df["asset_id"] = asset_id
    df["source_dataset"] = source_dataset
    df["selected_grid_lat"] = selected_grid_lat
    df["selected_grid_lon"] = selected_grid_lon
    df["extraction_note"] = extraction_note

    if var_map["u10"] in df.columns and var_map["v10"] in df.columns:
        df["wind_speed_10m"] = np.sqrt(df[var_map["u10"]] ** 2 + df[var_map["v10"]] ** 2)

    return df


def rename_and_convert(final_df, var_map_reference):
    """
    Harmonise variable names and convert raw units to modelling-friendly units.

    Temperature is converted from Kelvin to Celsius and precipitation from
    metres to millimetres to improve interpretability in later analysis.
    """
    rename_dict = {}
    if var_map_reference["temp"] is not None and var_map_reference["temp"] in final_df.columns:
        rename_dict[var_map_reference["temp"]] = "temp_2m_k"
    if var_map_reference["precip"] is not None and var_map_reference["precip"] in final_df.columns:
        rename_dict[var_map_reference["precip"]] = "precip_m"
    if var_map_reference["solar"] is not None and var_map_reference["solar"] in final_df.columns:
        rename_dict[var_map_reference["solar"]] = "solar_radiation_j_m2"
    if var_map_reference["u10"] is not None and var_map_reference["u10"] in final_df.columns:
        rename_dict[var_map_reference["u10"]] = "wind_u10"
    if var_map_reference["v10"] is not None and var_map_reference["v10"] in final_df.columns:
        rename_dict[var_map_reference["v10"]] = "wind_v10"

    final_df = final_df.rename(columns=rename_dict)

    if "temp_2m_k" in final_df.columns:
        final_df["temp_2m_c"] = final_df["temp_2m_k"] - 273.15
    if "precip_m" in final_df.columns:
        final_df["precip_mm"] = final_df["precip_m"] * 1000

    preferred_order = [
        "asset_id", "date", "temp_2m_k", "temp_2m_c", "precip_m", "precip_mm",
        "solar_radiation_j_m2", "wind_u10", "wind_v10", "wind_speed_10m",
        "source_dataset", "selected_grid_lat", "selected_grid_lon", "extraction_note",
    ]
    existing_cols = [c for c in preferred_order if c in final_df.columns]
    other_cols = [c for c in final_df.columns if c not in existing_cols]
    return final_df[existing_cols + other_cols]


def classify_asset(technology):
    """
    Assign each asset to the onshore or offshore extraction route.

    Offshore assets use ERA5. All others default to ERA5-Land because of its
    finer spatial resolution for terrestrial assets.
    """
    tech = str(technology).strip().lower()
    if any(k in tech for k in OFFSHORE_KEYWORDS):
        return "offshore"
    if any(k in tech for k in ONSHORE_KEYWORDS):
        return "onshore"
    return "onshore"


def build_climate_dataset(assets: pd.DataFrame):
    """
    Extract a monthly climate panel for every asset in the sample.

    ERA5-Land is used for onshore and solar assets because of its finer
    spatial resolution over land. ERA5 is used for offshore wind assets.
    Coastal assets whose nearest ERA5-Land cell is empty trigger a bounded
    radial search for the closest valid grid point. All extraction choices
    are recorded in the audit metadata for reproducibility.
    """
    print("Loading ERA5-Land...")
    ds_land = xr.open_dataset(ERA5_LAND_FILE)
    land_lat_name, land_lon_name, land_time_name, land_var_map = detect_names_and_vars(ds_land)

    print("Loading ERA5...")
    ds_era5 = xr.open_mfdataset(ERA5_FILES, combine="by_coords")
    era5_lat_name, era5_lon_name, era5_time_name, era5_var_map = detect_names_and_vars(ds_era5)

    rows = []
    issue_log = []

    for _, asset in assets.iterrows():
        asset_id = str(asset["asset_id"]).strip()
        technology = str(asset.get("technology", "")).strip()
        lat = float(asset["latitude"])
        lon = float(asset["longitude"])
        asset_class = classify_asset(technology)

        print(f"Processing climate for {asset_id} | {technology} | class={asset_class}")

        if asset_class == "offshore":
            lon_adj = normalise_longitude_for_dataset(lon, ds_era5, era5_lon_name)
            point = extract_nearest_point(ds_era5, lat, lon_adj, era5_lat_name, era5_lon_name)
            df = point_to_dataframe(point, era5_time_name, era5_var_map)
            selected_grid_lat = float(point[era5_lat_name].values)
            selected_grid_lon = float(point[era5_lon_name].values)
            extraction_note = "nearest_era5"
            source_dataset = "ERA5"

            if all_value_cols_nan(df, era5_time_name):
                extraction_note = "nearest_era5_all_nan"
                issue_log.append((asset_id, technology, extraction_note))
        else:
            lon_adj = normalise_longitude_for_dataset(lon, ds_land, land_lon_name)
            point = extract_nearest_point(ds_land, lat, lon_adj, land_lat_name, land_lon_name)
            df = point_to_dataframe(point, land_time_name, land_var_map)
            selected_grid_lat = float(point[land_lat_name].values)
            selected_grid_lon = float(point[land_lon_name].values)
            extraction_note = "nearest_era5_land"
            source_dataset = "ERA5-Land"

            if all_value_cols_nan(df, land_time_name):
                fallback_point, fb_lat, fb_lon, fb_note = extract_nearest_valid_land_point(
                    ds_land, lat, lon, land_lat_name, land_lon_name, land_time_name, land_var_map, max_steps=SEARCH_RADIUS_STEPS
                )
                if fallback_point is not None:
                    point = fallback_point
                    df = point_to_dataframe(point, land_time_name, land_var_map)
                    selected_grid_lat = fb_lat
                    selected_grid_lon = fb_lon
                    extraction_note = fb_note
                else:
                    extraction_note = fb_note
                    issue_log.append((asset_id, technology, extraction_note))

        used_time_name = era5_time_name if source_dataset == "ERA5" else land_time_name
        used_var_map = era5_var_map if source_dataset == "ERA5" else land_var_map

        df = standardise_extracted_df(
            df=df, asset_id=asset_id, source_dataset=source_dataset,
            selected_grid_lat=selected_grid_lat, selected_grid_lon=selected_grid_lon,
            extraction_note=extraction_note, time_name=used_time_name, var_map=used_var_map,
        )
        rows.append(df)

    final_df = pd.concat(rows, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.to_period("M").dt.to_timestamp()

    combined_var_map = land_var_map.copy()
    for k, v in era5_var_map.items():
        if combined_var_map.get(k) is None:
            combined_var_map[k] = v

    final_df = rename_and_convert(final_df, combined_var_map)
    final_df = final_df[(final_df["date"] >= START_DATE) & (final_df["date"] <= END_DATE)].copy()
    final_df.to_csv(CLIMATE_OUTPUT_FILE, index=False)

    climate_meta = {
        "output_file": str(CLIMATE_OUTPUT_FILE),
        "snapshot": dataset_snapshot(final_df, "climate_asset_monthly"),
        "extraction_note_counts": final_df["extraction_note"].value_counts(dropna=False).to_dict(),
        "unresolved_issues": issue_log,
    }
    return final_df, climate_meta


def detect_output_column(df):
    """
    Identify the generation column returned by Renewables.ninja.

    The API can label the output differently depending on endpoint and response
    format, so the detection logic remains intentionally flexible.
    """
    candidates = ["electricity", "output", "power", "generation"]
    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    exclude = {"time", "date", "local_time"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(f"Could not detect output column. Columns returned: {df.columns.tolist()}")


def monthly_from_ninja_mean_month(df, output_col):
    """
    Convert monthly mean power output into monthly generation per MW.

    Renewables.ninja returns monthly mean power. Multiplying by the number of
    hours in the month yields monthly energy generation normalised by 1 MW.
    """
    out = df.copy()
    date_col = "time" if "time" in out.columns else "date"
    out[date_col] = pd.to_datetime(out[date_col])
    out["date"] = out[date_col].dt.to_period("M").dt.to_timestamp()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["hours_in_month"] = out.apply(
        lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1] * 24,
        axis=1
    )
    out["generation_mwh_per_mw"] = out[output_col] * out["hours_in_month"] / 1000.0
    return out[["date", "generation_mwh_per_mw"]].copy()


def get_with_retries(session, url, params, max_retries=8):
    """
    Execute an API request with retry handling for rate limits and server errors.
    """
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after is not None else 60 * (attempt + 1)
            print(f"Rate limited (429). Waiting {wait_seconds} seconds before retrying...")
            time.sleep(wait_seconds)
            continue

        if r.status_code >= 500:
            wait_seconds = 30 * (attempt + 1)
            print(f"Server error {r.status_code}. Waiting {wait_seconds} seconds before retrying...")
            time.sleep(wait_seconds)
            continue

        r.raise_for_status()

    raise RuntimeError(f"Request failed after {max_retries} retries. URL: {url}")


def request_ninja_pv(session, lat, lon, date_from, date_to):
    """Request photovoltaic generation estimates for one location and year."""
    url = API_BASE + "data/pv"
    params = {
        "lat": float(lat), "lon": float(lon), "date_from": date_from, "date_to": date_to,
        "dataset": "merra2", "capacity": 1.0, "system_loss": 0.10, "tracking": 0,
        "tilt": 35, "azim": 180, "format": "json", "mean": "month", "raw": "false",
    }
    r = get_with_retries(session, url, params)
    payload = r.json()
    if "data" not in payload:
        raise ValueError(f"Unexpected PV response: {payload}")
    df = pd.DataFrame.from_dict(payload["data"], orient="index").reset_index()
    return df.rename(columns={"index": "time"})


def request_ninja_wind(session, lat, lon, date_from, date_to, technology):
    """
    Request wind generation estimates for one location and year.

    Turbine specification and hub height vary for onshore and offshore assets to
    preserve a plausible technology-specific parameterisation.
    """
    url = API_BASE + "data/wind"
    tech = str(technology).strip().lower()
    if tech == "offshore_wind":
        height = 120
        turbine = "Vestas V90 3000"
    else:
        height = 100
        turbine = "Vestas V80 2000"

    params = {
        "lat": float(lat), "lon": float(lon), "date_from": date_from, "date_to": date_to,
        "dataset": "merra2", "capacity": 1.0, "height": height, "turbine": turbine,
        "format": "json", "mean": "month", "raw": "false",
    }
    r = get_with_retries(session, url, params)
    payload = r.json()
    if "data" not in payload:
        raise ValueError(f"Unexpected wind response: {payload}")
    df = pd.DataFrame.from_dict(payload["data"], orient="index").reset_index()
    return df.rename(columns={"index": "time"})


def load_assets_for_generation(path):
    """
    Load and validate the subset of asset metadata needed for API generation calls.
    """
    assets = pd.read_csv(path)
    required_cols = ["asset_id", "asset_name", "technology", "latitude", "longitude", "capacity_mw"]
    missing = [c for c in required_cols if c not in assets.columns]
    if missing:
        raise ValueError(f"assets.csv is missing required columns: {missing}")

    assets["latitude"] = pd.to_numeric(assets["latitude"], errors="coerce")
    assets["longitude"] = pd.to_numeric(assets["longitude"], errors="coerce")
    assets["capacity_mw"] = pd.to_numeric(assets["capacity_mw"], errors="coerce")

    if "ownership_pct" in assets.columns:
        assets["ownership_fraction"] = assets["ownership_pct"].apply(clean_percentage)
    else:
        assets["ownership_fraction"] = 1.0

    assets["ownership_fraction"] = pd.to_numeric(assets["ownership_fraction"], errors="coerce").fillna(1.0)
    assets = assets.dropna(subset=["asset_id", "technology", "latitude", "longitude", "capacity_mw"]).copy()

    tech_allowed = {"solar", "onshore_wind", "offshore_wind"}
    assets = assets[assets["technology"].isin(tech_allowed)].copy()

    if TEST_ASSET_LIMIT is not None:
        assets = assets.head(TEST_ASSET_LIMIT).copy()
    return assets


def load_existing_progress(progress_file):
    """Resume partially completed generation downloads from a saved checkpoint."""
    if progress_file.exists():
        df = pd.read_csv(progress_file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def save_progress(df, progress_file):
    """Persist intermediate generation results after each completed asset."""
    df.to_csv(progress_file, index=False)


def fetch_asset_generation(session, asset_row):
    """
    Download annual generation estimates for one asset and combine them into a
    single monthly time series covering the full empirical window.
    """
    asset_id = asset_row["asset_id"]
    asset_name = asset_row["asset_name"]
    technology = asset_row["technology"]
    lat = asset_row["latitude"]
    lon = asset_row["longitude"]
    capacity_mw = asset_row["capacity_mw"]
    ownership_fraction = asset_row["ownership_fraction"]

    monthly_parts = []
    for year in range(START_YEAR, END_YEAR + 1):
        date_from = f"{year}-01-01"
        date_to = f"{year}-12-31"
        print(f"Requesting generation for {asset_id} | {technology} | {year}")

        if technology == "solar":
            raw_df = request_ninja_pv(session, lat, lon, date_from, date_to)
        else:
            raw_df = request_ninja_wind(session, lat, lon, date_from, date_to, technology)

        output_col = detect_output_column(raw_df)
        monthly_df = monthly_from_ninja_mean_month(raw_df, output_col)

        monthly_df["asset_id"] = asset_id
        monthly_df["asset_name"] = asset_name
        monthly_df["technology"] = technology
        monthly_df["latitude"] = lat
        monthly_df["longitude"] = lon
        monthly_df["capacity_mw"] = capacity_mw
        monthly_df["ownership_fraction"] = ownership_fraction
        monthly_df["generation_mwh_est"] = monthly_df["generation_mwh_per_mw"] * capacity_mw
        monthly_df["generation_mwh_est_owned"] = monthly_df["generation_mwh_est"] * ownership_fraction

        monthly_parts.append(monthly_df)
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    return pd.concat(monthly_parts, ignore_index=True)


def build_generation_dataset():
    """
    Build the portfolio-wide generation dataset from Renewables.ninja outputs.

    Long-running API collection is made robust through checkpointing so that
    successful assets do not need to be downloaded again after interruption.
    """
    if TOKEN == "PASTE_YOUR_NEW_RENEWABLES_NINJA_TOKEN_HERE":
        raise ValueError("Please paste your Renewables.ninja token into the TOKEN variable first.")

    assets = load_assets_for_generation(ASSETS_FILE)
    session = requests.Session()
    session.headers = {"Authorization": f"Token {TOKEN}"}

    existing = load_existing_progress(GENERATION_PROGRESS_FILE)
    completed_ids = set(existing["asset_id"].unique()) if existing is not None else set()

    all_results = []
    if existing is not None:
        all_results.append(existing)

    for _, asset in assets.iterrows():
        asset_id = asset["asset_id"]
        if asset_id in completed_ids:
            print(f"Skipping {asset_id} (already in progress file)")
            continue

        asset_result = fetch_asset_generation(session, asset)
        all_results.append(asset_result)

        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(["asset_id", "date"]).reset_index(drop=True)
        save_progress(combined, GENERATION_PROGRESS_FILE)

    final_generation = pd.concat(all_results, ignore_index=True)
    final_generation["date"] = pd.to_datetime(final_generation["date"])
    final_generation = final_generation[(final_generation["date"] >= START_DATE) & (final_generation["date"] <= END_DATE)].copy()
    final_generation = final_generation.sort_values(["asset_id", "date"]).reset_index(drop=True)
    final_generation.to_csv(GENERATION_OUTPUT_FILE, index=False)

    generation_meta = {
        "output_file": str(GENERATION_OUTPUT_FILE),
        "snapshot": dataset_snapshot(final_generation, "generation_monthly"),
        "progress_file": str(GENERATION_PROGRESS_FILE),
    }
    return final_generation, generation_meta


def build_prices_dataset():
    """
    Download BMRS market index prices and aggregate them to monthly frequency.

    The API is queried in seven-day windows because of source-side constraints.
    The resulting high-frequency prices are averaged to monthly values to align
    with the rest of the modelling pipeline.
    """
    all_chunks = []
    current_start = BMRS_START

    print("Starting BMRS download...")
    while current_start <= BMRS_END:
        current_end = min(current_start + timedelta(days=6, hours=23, minutes=59), BMRS_END)
        params = {
            "from": current_start.strftime("%Y-%m-%dT%H:%MZ"),
            "to": current_end.strftime("%Y-%m-%dT%H:%MZ"),
            "format": "json"
        }
        r = requests.get(BMRS_BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "data" in data:
            df_chunk = pd.DataFrame(data["data"])
        elif isinstance(data, list):
            df_chunk = pd.DataFrame(data)
        else:
            df_chunk = pd.DataFrame(data)

        all_chunks.append(df_chunk)
        current_start = current_end + timedelta(minutes=1)

    df = pd.concat(all_chunks, ignore_index=True).drop_duplicates()

    time_candidates = ["startTime", "settlementDate", "settlementPeriodStart", "timestamp", "date"]
    price_candidates = ["marketIndexPrice", "MarketIndexPrice", "price", "Price", "midPrice"]

    time_col = next((c for c in time_candidates if c in df.columns), None)
    price_col = next((c for c in price_candidates if c in df.columns), None)

    if time_col is None or price_col is None:
        raise ValueError(f"Could not identify time/price columns. Returned columns: {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[time_col, price_col]).copy()

    prices_monthly = (
        df.assign(date=lambda x: x[time_col].dt.to_period("M").dt.to_timestamp())
          .groupby("date", as_index=False)[price_col]
          .mean()
          .rename(columns={price_col: "power_price_gbp_mwh"})
    )
    prices_monthly = prices_monthly.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    prices_monthly = prices_monthly[(prices_monthly["date"] >= START_DATE) & (prices_monthly["date"] <= END_DATE)].copy()
    prices_monthly.to_csv(PRICES_OUTPUT_FILE, index=False)

    prices_meta = {
        "output_file": str(PRICES_OUTPUT_FILE),
        "snapshot": dataset_snapshot(prices_monthly, "prices_monthly"),
    }
    return prices_monthly, prices_meta


def check_duplicates(df, keys, name):
    """Report duplicate key combinations before panel merging."""
    dupes = int(df.duplicated(subset=keys).sum())
    print(f"{name} duplicate rows on {keys}: {dupes}")
    return dupes


def build_model_panel():
    """
    Merge climate, generation, price, and asset metadata into the final panel.

    Generation is used as the merge anchor, so only asset-months present in
    the generation dataset enter the panel. The cleaned output additionally
    excludes pre-operational periods and observations missing any required
    modelling variable.
    """
    assets = pd.read_csv(ASSETS_FILE)
    climate = pd.read_csv(CLIMATE_OUTPUT_FILE)
    generation = pd.read_csv(GENERATION_OUTPUT_FILE)
    prices = pd.read_csv(PRICES_OUTPUT_FILE)

    assets = assets.dropna(how="all").copy()
    climate = climate.dropna(how="all").copy()
    generation = generation.dropna(how="all").copy()
    prices = prices.dropna(how="all").copy()

    for df_name, df in [("assets", assets), ("climate", climate), ("generation", generation), ("prices", prices)]:
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
            print(f"Dropped unnamed columns from {df_name}: {unnamed_cols}")

    assets = assets.dropna(subset=["asset_id"]).copy()
    climate = climate.dropna(subset=["asset_id", "date"]).copy()
    generation = generation.dropna(subset=["asset_id", "date"]).copy()
    prices = prices.dropna(subset=["date"]).copy()

    for df in [climate, generation, prices]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    assets["start_of_operations"] = pd.to_datetime(assets["start_of_operations"], errors="coerce", dayfirst=True)

    for df in [assets, climate, generation]:
        df["asset_id"] = df["asset_id"].astype(str).str.strip()

    climate = climate[(climate["date"] >= START_DATE) & (climate["date"] <= END_DATE)].copy()
    generation = generation[(generation["date"] >= START_DATE) & (generation["date"] <= END_DATE)].copy()
    prices = prices[(prices["date"] >= START_DATE) & (prices["date"] <= END_DATE)].copy()

    climate_dupes = check_duplicates(climate, ["asset_id", "date"], "climate")
    generation_dupes = check_duplicates(generation, ["asset_id", "date"], "generation")
    prices_dupes = check_duplicates(prices, ["date"], "prices")

    if climate_dupes > 0:
        climate_num_cols = climate.select_dtypes(include=np.number).columns.tolist()
        climate = climate.groupby(["asset_id", "date"], as_index=False)[climate_num_cols].mean()

    if prices_dupes > 0:
        prices = prices.groupby("date", as_index=False)["power_price_gbp_mwh"].mean()

    assets_extra = assets[
        [
            "asset_id", "country", "ownership_pct", "operational_status", "start_of_operations",
            "commissioning_year", "portfolio_capacity_pct", "remaining_asset_life_years",
        ]
    ].copy()

    # Generation is used as the merge anchor so that only asset-months with
    # a valid generation estimate enter the panel. Left joins preserve that
    # structure while allowing climate, asset, or price data to be missing.
    panel = (
        generation
        .merge(climate, on=["asset_id", "date"], how="left", validate="many_to_one")
        .merge(assets_extra, on="asset_id", how="left", validate="many_to_one")
        .merge(prices, on="date", how="left", validate="many_to_one")
    )

    # Normalising by capacity makes the forecasting target comparable across
    # assets of different sizes and decouples the modelling stage from the
    # financial scaling applied downstream.
    panel["generation_per_mw"] = panel["generation_mwh_est_owned"] / panel["capacity_mw"]
    panel["revenue_proxy_gbp"] = panel["generation_mwh_est_owned"] * panel["power_price_gbp_mwh"]
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month

    panel["pre_operational_flag"] = 0
    mask_preop = panel["start_of_operations"].notna() & (panel["date"] < panel["start_of_operations"])
    panel.loc[mask_preop, "pre_operational_flag"] = 1

    panel = panel.sort_values(["asset_id", "date"]).reset_index(drop=True)

    # All lags and rolling statistics are computed within asset groups and
    # shifted by at least one period to prevent data leakage at training time.
    lag_cols = ["temp_2m_c", "precip_mm", "solar_radiation_j_m2", "wind_speed_10m", "power_price_gbp_mwh"]
    for c in lag_cols:
        panel[f"{c}_lag1"] = panel.groupby("asset_id")[c].shift(1)

    rolling_cols = ["temp_2m_c", "precip_mm", "solar_radiation_j_m2", "wind_speed_10m"]
    for c in rolling_cols:
        panel[f"{c}_roll3_mean"] = (
            panel.groupby("asset_id")[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    for c in rolling_cols:
        month_avg = panel.groupby(["asset_id", "month"])[c].transform("mean")
        panel[f"{c}_anomaly"] = panel[c] - month_avg

    panel.to_csv(MODEL_PANEL_OUTPUT_FILE, index=False)

    # Pre-operational observations are excluded because generation in those
    # periods does not represent the asset's operating regime and would bias
    # the seasonal baselines used as benchmark comparators.
    model_panel_clean = panel.copy()
    model_panel_clean = model_panel_clean[model_panel_clean["capacity_mw"] > 0].copy()
    model_panel_clean = model_panel_clean[model_panel_clean["pre_operational_flag"] == 0].copy()

    required_cols = [
        "generation_per_mw", "generation_mwh_est_owned", "power_price_gbp_mwh",
        "temp_2m_c", "precip_mm", "solar_radiation_j_m2", "wind_speed_10m",
    ]
    model_panel_clean = model_panel_clean.dropna(subset=required_cols).copy()
    model_panel_clean = model_panel_clean.sort_values(["asset_id", "date"]).reset_index(drop=True)
    model_panel_clean.to_csv(MODEL_PANEL_CLEAN_OUTPUT_FILE, index=False)

    merge_meta = {
        "output_file_panel": str(MODEL_PANEL_OUTPUT_FILE),
        "output_file_panel_clean": str(MODEL_PANEL_CLEAN_OUTPUT_FILE),
        "panel_snapshot": dataset_snapshot(panel, "model_panel"),
        "panel_clean_snapshot": dataset_snapshot(model_panel_clean, "model_panel_clean"),
        "missing_top_20": panel.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        "climate_duplicate_rows_before_fix": climate_dupes,
        "generation_duplicate_rows_before_fix": generation_dupes,
        "prices_duplicate_rows_before_fix": prices_dupes,
        "feature_examples": {
            "core_climate_features": ["temp_2m_c", "precip_mm", "solar_radiation_j_m2", "wind_speed_10m"],
            "key_engineered_features": [
                "generation_per_mw", "revenue_proxy_gbp", "temp_2m_c_lag1",
                "wind_speed_10m_roll3_mean", "solar_radiation_j_m2_anomaly"
            ],
        }
    }
    return panel, model_panel_clean, merge_meta


def main():
    """
    Execute the full data build pipeline. Stage flags in the module header
    control which steps run; the JSON audit file records dimensional summaries
    from each stage for reproducibility checking.
    """
    summary = {"inputs": {"assets_file": str(ASSETS_FILE)}, "steps": {}}

    assets = pd.read_csv(ASSETS_FILE)
    assets = assets.dropna(subset=["asset_id", "latitude", "longitude"]).copy()
    summary["steps"]["assets_input"] = dataset_snapshot(assets, "assets")

    if RUN_CLIMATE:
        _, climate_meta = build_climate_dataset(assets)
        summary["steps"]["climate_build"] = climate_meta

    if RUN_GENERATION:
        _, generation_meta = build_generation_dataset()
        summary["steps"]["generation_build"] = generation_meta

    if RUN_PRICES:
        _, prices_meta = build_prices_dataset()
        summary["steps"]["prices_build"] = prices_meta

    if RUN_MERGE:
        _, _, merge_meta = build_model_panel()
        summary["steps"]["merge_build"] = merge_meta

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================")
    print("Data build complete.")
    print(f"Saved: {CLIMATE_OUTPUT_FILE.name}")
    print(f"Saved: {GENERATION_OUTPUT_FILE.name}")
    print(f"Saved: {PRICES_OUTPUT_FILE.name}")
    print(f"Saved: {MODEL_PANEL_OUTPUT_FILE.name}")
    print(f"Saved: {MODEL_PANEL_CLEAN_OUTPUT_FILE.name}")
    print(f"Saved: {SUMMARY_FILE.name}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
