"""Clean + transform + prepare: imputation, outlier removal and feature engineering."""

import numpy as np
import pandas as pd

from .config import VESSEL_CLASSES

NUMERIC_FEATURES = [
    "capacity_t", "design_speed_kn", "mcr_kw", "sfoc_g_kwh", "aux_kw", "age_yr",
    "speed_kn", "load_frac", "distance_nm", "wind_kn", "wave_m", "current_kn",
    # engineered
    "speed_ratio", "speed_ratio_cubed", "sailing_hr_est", "power_proxy_mwh", "sea_state_index",
]
TYPE_COLUMNS = [f"type_{c}" for c in VESSEL_CLASSES]
FEATURES = NUMERIC_FEATURES + TYPE_COLUMNS


def clean_history(history: pd.DataFrame):
    """Impute missing weather and drop implausible fuel records. Returns (clean_df, report)."""
    df = history.copy()
    report = {"rows_in": len(df)}
    missing = df[["wind_kn", "wave_m", "current_kn"]].isna().sum()
    report["missing_imputed"] = int(missing.sum())
    for col in ["wind_kn", "wave_m", "current_kn"]:
        df[col] = df[col].fillna(df[col].median())

    # Outliers: fuel intensity (t per MWh of estimated work) outside robust IQR fence.
    feats = add_features(df)
    intensity = df["fuel_t"] / feats["power_proxy_mwh"].clip(lower=1e-3)
    q1, q3 = intensity.quantile([0.25, 0.75])
    fence = q3 + 3 * (q3 - q1)
    keep = (intensity <= fence) & (df["fuel_t"] > 0)
    report["outliers_removed"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speed_ratio"] = out["speed_kn"] / out["design_speed_kn"]
    out["speed_ratio_cubed"] = out["speed_ratio"] ** 3
    out["sailing_hr_est"] = out["distance_nm"] / np.maximum(out["speed_kn"] + out["current_kn"], 3.0)
    out["power_proxy_mwh"] = out["mcr_kw"] * out["speed_ratio_cubed"] * out["sailing_hr_est"] / 1000
    out["sea_state_index"] = 0.010 * out["wind_kn"] + 0.06 * np.power(out["wave_m"].clip(lower=0), 1.5)
    for cls in VESSEL_CLASSES:
        out[f"type_{cls}"] = (out["vessel_type"] == cls).astype(int)
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)[FEATURES]
