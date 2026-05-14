"""
preprocessing.py
================
Data preprocessing and exploratory data analysis (EDA) for the
Payroll Anomaly Detection System.

Responsibilities:
  - Load payroll CSV datasets
  - Handle missing values via median imputation
  - Drop non-feature columns (employee_id, department, labels)
  - Normalize / scale numeric features with StandardScaler
  - Generate EDA visualizations (correlation heatmap, box plots, distributions)
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Columns that are metadata / labels — excluded from ML training features
META_COLUMNS = ["employee_id", "department", "is_anomaly", "anomaly_type"]

NUMERIC_FEATURES = [
    "basic_salary", "hra", "other_allowances",
    "overtime_hours", "bonuses", "tax_deductions",
    "other_deductions", "net_salary",
    "years_of_service", "performance_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a payroll CSV and perform basic sanity-checks.

    Parameters
    ----------
    filepath : Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame with raw payroll data.

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    ValueError if required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Check that at least the core numeric columns are present
    required = {"basic_salary", "hra", "net_salary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"[preprocessing] Loaded {len(df)} records, {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Missing Value Handling
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values with column medians.
    Categorical columns get 'Unknown' fill.

    Parameters
    ----------
    df : Raw payroll DataFrame.

    Returns
    -------
    DataFrame with no missing values.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    # Numeric: median imputation (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    # Categorical: fill with 'Unknown'
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna("Unknown", inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"[preprocessing] Missing values: {missing_before} → {missing_after}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric features relevant to anomaly detection.
    Drops metadata/label columns. Adds derived features.

    Parameters
    ----------
    df : Clean payroll DataFrame.

    Returns
    -------
    DataFrame containing only numeric training features.
    """
    df = df.copy()

    # ── Derived features ─────────────────────────────────────────────────────
    # Ratio: net salary vs gross (should be 0.6–0.95 for a normal employee)
    gross = df["basic_salary"] + df.get("hra", 0) + df.get("other_allowances", 0) \
            + df.get("bonuses", 0)
    gross = gross.replace(0, np.nan)
    df["net_to_gross_ratio"] = (df["net_salary"] / gross).fillna(0)

    # Ratio: tax deductions vs basic salary (should be roughly 0.10–0.30)
    df["tax_rate"] = (df["tax_deductions"] / df["basic_salary"].replace(0, np.nan)).fillna(0)

    # Ratio: bonuses vs basic salary
    df["bonus_rate"] = (df["bonuses"] / df["basic_salary"].replace(0, np.nan)).fillna(0)

    # Overtime flag: > 100 hours is physically suspicious
    df["high_overtime_flag"] = (df["overtime_hours"] > 100).astype(int)

    # Columns to keep for training
    feature_cols = NUMERIC_FEATURES + [
        "net_to_gross_ratio", "tax_rate", "bonus_rate", "high_overtime_flag"
    ]

    # Only keep columns that actually exist in the dataframe
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    # Clip extreme values: cap at 99.5th percentile to reduce extreme skew
    for col in X.select_dtypes(include=[np.number]).columns:
        upper = X[col].quantile(0.995)
        lower = X[col].quantile(0.005)
        X[col] = X[col].clip(lower=lower, upper=upper)

    print(f"[preprocessing] Feature matrix shape: {X.shape}")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(X: pd.DataFrame, scaler: StandardScaler | None = None,
                   scaler_save_path: str | None = None
                   ) -> tuple[np.ndarray, StandardScaler]:
    """
    Fit (or apply) a StandardScaler to the feature matrix.

    Parameters
    ----------
    X                : Feature DataFrame.
    scaler           : Pre-fitted scaler to reuse. If None, a new one is fitted.
    scaler_save_path : If provided, saves the scaler to this path.

    Returns
    -------
    (X_scaled: np.ndarray, scaler: StandardScaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"[preprocessing] Scaler fitted on {X.shape[1]} features.")
    else:
        X_scaled = scaler.transform(X)
        print("[preprocessing] Applied existing scaler.")

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"[preprocessing] Scaler saved → {scaler_save_path}")

    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(filepath: str,
                         scaler_save_path: str = "models/scaler.pkl"
                         ) -> tuple[np.ndarray, pd.DataFrame, StandardScaler, list]:
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    filepath         : Path to the payroll CSV.
    scaler_save_path : Where to persist the fitted scaler.

    Returns
    -------
    (X_scaled, df_clean, scaler, feature_names)
    """
    df = load_data(filepath)
    df = handle_missing_values(df)
    X  = extract_features(df)
    X_scaled, scaler = scale_features(X, scaler_save_path=scaler_save_path)
    return X_scaled, df, scaler, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# EDA Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def eda_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Return a Plotly heatmap showing feature correlation."""
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["is_anomaly"] if c in df.columns], errors="ignore"
    )
    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
        template="plotly_dark",
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def eda_box_plots(df: pd.DataFrame, features: list | None = None) -> go.Figure:
    """
    Return a Plotly figure with box plots for each numeric feature.
    If the dataset has an 'is_anomaly' column, normal vs anomalous distributions
    are shown side-by-side.
    """
    if features is None:
        features = NUMERIC_FEATURES

    has_label = "is_anomaly" in df.columns
    fig = go.Figure()

    palette = {"Normal": "#4FC3F7", "Anomaly": "#FF6B6B"}

    for feat in features:
        if feat not in df.columns:
            continue
        if has_label:
            for label_val, label_name in [(0, "Normal"), (1, "Anomaly")]:
                subset = df[df["is_anomaly"] == label_val][feat].dropna()
                fig.add_trace(go.Box(
                    y=subset, name=f"{label_name} — {feat}",
                    marker_color=palette[label_name],
                    boxmean=True,
                    legendgroup=label_name,
                    showlegend=(feat == features[0]),
                ))
        else:
            fig.add_trace(go.Box(
                y=df[feat].dropna(), name=feat,
                marker_color="#4FC3F7", boxmean=True))

    fig.update_layout(
        title="Feature Distributions (Box Plots)",
        yaxis_title="Value",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=60),
        showlegend=has_label,
    )
    return fig


def eda_histograms(df: pd.DataFrame, feature: str) -> go.Figure:
    """Return a histogram for a single feature, colored by anomaly label if available."""
    has_label = "is_anomaly" in df.columns

    if has_label:
        fig = px.histogram(
            df, x=feature,
            color=df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            nbins=50, barmode="overlay", opacity=0.75,
            color_discrete_map={"Normal": "#4FC3F7", "Anomaly": "#FF6B6B"},
            title=f"Distribution: {feature}",
            template="plotly_dark",
        )
    else:
        fig = px.histogram(
            df, x=feature, nbins=50,
            title=f"Distribution: {feature}",
            template="plotly_dark",
        )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def eda_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return formatted descriptive statistics for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe().T
    stats["missing"] = df[num_df.columns].isnull().sum()
    stats["missing_%"] = (stats["missing"] / len(df) * 100).round(2)
    return stats.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_scaled, df, scaler, feats = preprocess_pipeline("data/payroll_dataset.csv")
    print(f"Feature names: {feats}")
    print(f"Scaled shape: {X_scaled.shape}")
    print(eda_summary_stats(df).to_string())
