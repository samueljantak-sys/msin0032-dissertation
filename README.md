# MSIN0032 Dissertation Code and Data

This repository contains the core code, input data, processed datasets, and figures used in my MSIN0032 Management Science Dissertation.

## Contents

- `build_model_panel.py` — constructs the integrated modelling panel from asset, climate, generation, and price data
- `model_suite.py` — runs baseline and machine-learning forecasting models and produces model evaluation outputs
- `financial_layer.py` — converts generation forecasts into revenue and portfolio-level risk metrics
- `assets.csv` — master asset file
- `climate_asset_monthly.csv` — processed climate panel
- `generation_monthly.csv` — processed generation estimates
- `prices_monthly.csv` — processed monthly price series
- `model_panel.csv` — merged full panel
- `model_panel_clean.csv` — cleaned modelling panel
- `Figures/` — figures used in the dissertation

## API Access

The generation data script requires a Renewables.ninja API token.  
Users should obtain their own token and insert it in the script before running.

## Requirements

The analysis was conducted in Python using standard data science libraries. Key dependencies are listed in `requirements.txt`.
  
## Notes

This repository contains the core materials used for the final dissertation submission. Intermediate comparison scripts and large derived output folders were excluded for clarity.
