"""
train.py
========
Model training, evaluation, and explainability for the
Payroll Anomaly Detection System.

Models
------
  1. Isolation Forest  (primary)
  2. One-Class SVM     (comparison)
  3. Local Outlier Factor (comparison)
  4. Ensemble voting   (majority rule across the three)

Outputs
-------
  - Trained model files persisted to models/
  - data/anomaly_results.csv  —  full dataset with anomaly labels & scores
  - Console summary: anomaly count, %, per-model comparison table
"""

import os
import time
import warnings
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_pipeline, extract_features, scale_features, load_data, handle_missing_values

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
DATA_DIR    = "data"
RESULTS_CSV = os.path.join(DATA_DIR, "anomaly_results.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_models(contamination: float = 0.08) -> dict:
    """
    Instantiate all three anomaly detection models.

    Parameters
    ----------
    contamination : Expected proportion of anomalies (0.01 – 0.5).
                    Used by Isolation Forest and LOF.

    Returns
    -------
    Dict mapping model name → unfitted model instance.
    """
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        ),
        "One-Class SVM": OneClassSVM(
            kernel="rbf",
            nu=contamination,           # nu ≈ upper bound on fraction of outliers
            gamma="scale",
            cache_size=500,
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=False,              # transductive mode (fit+predict together)
            algorithm="auto",
            n_jobs=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_scaled: np.ndarray,
                 contamination: float = 0.08,
                 save_models: bool = True
                 ) -> tuple[dict, dict, dict]:
    """
    Fit all anomaly detection models and return predictions + scores.

    Parameters
    ----------
    X_scaled      : Scaled numpy feature array.
    contamination : Outlier fraction parameter.
    save_models   : Whether to persist fitted models to disk.

    Returns
    -------
    (models, predictions, scores)

    predictions : dict  model_name → np.ndarray of {-1=anomaly, 1=normal}
    scores      : dict  model_name → np.ndarray of floating-point anomaly score
                        (higher = more anomalous, unified direction)
    """
    models = build_models(contamination)
    predictions = {}
    scores      = {}
    times       = {}

    print(f"\n{'─'*60}")
    print(f"  Training Anomaly Detection Models  (contamination={contamination:.2%})")
    print(f"{'─'*60}")

    for name, model in models.items():
        t0 = time.time()

        if name == "Local Outlier Factor":
            # LOF transductive: fit_predict in one call
            raw_preds = model.fit_predict(X_scaled)
            # decision_function not available when novelty=False; use negative_outlier_factor_
            raw_scores = -model.negative_outlier_factor_   # higher = more anomalous

        elif name == "One-Class SVM":
            model.fit(X_scaled)
            raw_preds  = model.predict(X_scaled)
            # decision_function: negative inside, positive outside the frontier
            raw_scores = -model.decision_function(X_scaled)  # flip sign: higher = more anomalous

        else:  # Isolation Forest
            model.fit(X_scaled)
            raw_preds  = model.predict(X_scaled)
            raw_scores = -model.score_samples(X_scaled)      # higher = more anomalous

        elapsed = time.time() - t0

        # Normalize scores to [0, 1] range for comparability
        s_min, s_max = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - s_min) / (s_max - s_min + 1e-9)

        predictions[name] = raw_preds      # -1 = anomaly, 1 = normal
        scores[name]      = norm_scores
        times[name]       = elapsed

        n_anomalies  = (raw_preds == -1).sum()
        pct_anomalies = n_anomalies / len(raw_preds) * 100
        print(f"  [{name:25s}]  anomalies={n_anomalies:4d} ({pct_anomalies:5.1f}%)  "
              f"time={elapsed:.2f}s")

        if save_models and name != "Local Outlier Factor":
            # LOF in transductive mode cannot be saved for re-use on new data
            path = os.path.join(MODELS_DIR, name.lower().replace(" ", "_").replace("-", "") + ".pkl")
            joblib.dump(model, path)
            print(f"    └─ saved → {path}")

    print(f"{'─'*60}\n")
    return models, predictions, scores


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(predictions: dict, threshold: int = 2) -> np.ndarray:
    """
    Majority-vote ensemble: flag a record as anomalous if ≥ `threshold` models agree.

    Parameters
    ----------
    predictions : Dict from train_models — model_name → raw sklearn predictions.
    threshold   : Minimum number of models that must agree (default 2 of 3).

    Returns
    -------
    np.ndarray of {0=normal, 1=anomaly}
    """
    # Convert sklearn convention (-1=anomaly, 1=normal) → (1=anomaly, 0=normal)
    votes = np.stack([
        (preds == -1).astype(int) for preds in predictions.values()
    ], axis=1)   # shape: (n_samples, n_models)

    vote_sum = votes.sum(axis=1)
    return (vote_sum >= threshold).astype(int)


def ensemble_score(scores: dict) -> np.ndarray:
    """Average the normalized anomaly scores across all models."""
    score_matrix = np.stack(list(scores.values()), axis=1)
    return score_matrix.mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance(X_df: pd.DataFrame,
                                labels: np.ndarray,
                                scaler: StandardScaler
                                ) -> np.ndarray:
    """
    Compute per-feature importance by measuring how much anomalous records deviate
    from normal records in the original (unscaled) feature space.

    This is a model-agnostic approach that doesn't require SHAP, making it fast
    and universally applicable across all three models.

    Parameters
    ----------
    X_df   : Original (unscaled) feature DataFrame.
    labels : Ensemble anomaly labels (0=normal, 1=anomaly).
    scaler : Fitted StandardScaler (used to get feature means).

    Returns
    -------
    np.ndarray of shape (n_features,) — mean absolute deviation of anomalies
    from the normal population mean, normalized to [0, 1].
    """
    normal_mask  = labels == 0
    anomaly_mask = labels == 1

    if anomaly_mask.sum() == 0:
        return np.zeros(X_df.shape[1])

    normal_means  = X_df[normal_mask].mean()
    anomaly_means = X_df[anomaly_mask].mean()
    normal_stds   = X_df[normal_mask].std().replace(0, 1)

    # Standardized deviation
    deviations = ((anomaly_means - normal_means) / normal_stds).abs()
    deviations_arr = deviations.values

    # Normalize to [0, 1]
    d_max = deviations_arr.max()
    if d_max > 0:
        deviations_arr = deviations_arr / d_max

    return deviations_arr


def explain_record(record: pd.Series, X_df: pd.DataFrame,
                   labels: np.ndarray) -> pd.DataFrame:
    """
    Explain why a single record is flagged as anomalous.

    For each feature, compute Z-score vs the normal population.

    Parameters
    ----------
    record : A single-row Series from the original (unscaled) DataFrame.
    X_df   : Full original feature DataFrame.
    labels : Ensemble anomaly labels.

    Returns
    -------
    DataFrame with columns [feature, normal_mean, record_value, z_score, flagged]
    sorted by |z_score| descending.
    """
    normal_mask = labels == 0
    normal_df   = X_df[normal_mask]

    rows = []
    for col in X_df.columns:
        if col not in record.index:
            continue
        mu  = normal_df[col].mean()
        sig = normal_df[col].std()
        val = record[col]
        z   = (val - mu) / (sig + 1e-9) if sig > 0 else 0.0
        rows.append({
            "Feature":      col,
            "Normal Mean":  round(mu, 2),
            "Record Value": round(val, 2),
            "Z-Score":      round(z, 3),
            "Flagged":      abs(z) > 3,
        })

    result = pd.DataFrame(rows)
    result = result.reindex(result["Z-Score"].abs().sort_values(ascending=False).index)
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Results Assembly + Saving
# ─────────────────────────────────────────────────────────────────────────────

def assemble_results(df_clean: pd.DataFrame,
                     predictions: dict,
                     scores: dict,
                     ensemble_labels: np.ndarray,
                     ens_scores: np.ndarray,
                     feature_importances: np.ndarray,
                     feature_names: list,
                     save_path: str = RESULTS_CSV) -> pd.DataFrame:
    """
    Attach all model predictions, scores, and ensemble results to the original
    DataFrame, then save to CSV.

    Returns the augmented DataFrame.
    """
    results = df_clean.copy()

    # Per-model columns
    for model_name, preds in predictions.items():
        col = model_name.lower().replace(" ", "_").replace("-", "")
        results[f"{col}_pred"]  = (preds == -1).astype(int)
        results[f"{col}_score"] = scores[model_name].round(6)

    # Ensemble columns
    results["ensemble_anomaly"] = ensemble_labels
    results["ensemble_score"]   = ens_scores.round(6)

    # Top anomalous feature per record (highest absolute contribution)
    if len(feature_names) == len(feature_importances):
        top_feature_idx = int(np.argmax(feature_importances))
        results["top_anomaly_feature"] = feature_names[top_feature_idx]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results.to_csv(save_path, index=False)
    print(f"[train] Results saved → {save_path}  ({len(results)} records)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(predictions: dict,
                            ground_truth: np.ndarray | None = None
                            ) -> pd.DataFrame:
    """
    Build a model comparison DataFrame.

    If `ground_truth` is provided (0/1 labels), compute precision/recall/F1.
    Otherwise, report detection count and rate only.
    """
    rows = []
    for name, preds in predictions.items():
        detected = (preds == -1).sum()
        rate     = detected / len(preds) * 100

        row = {
            "Model":             name,
            "Anomalies Detected": int(detected),
            "Detection Rate (%)": round(rate, 2),
        }

        if ground_truth is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score
            binary_preds = (preds == -1).astype(int)
            row["Precision"] = round(precision_score(ground_truth, binary_preds,
                                                      zero_division=0), 4)
            row["Recall"]    = round(recall_score(ground_truth, binary_preds,
                                                   zero_division=0), 4)
            row["F1-Score"]  = round(f1_score(ground_truth, binary_preds,
                                               zero_division=0), 4)
        rows.append(row)

    # Ensemble row
    ens_detected = (predictions.get("Isolation Forest", np.array([])) == -1).sum()
    rows.append({
        "Model": "Ensemble (Majority Vote)",
        "Anomalies Detected": "—",
        "Detection Rate (%)": "—",
    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(data_path: str = "data/payroll_dataset.csv",
                           contamination: float = 0.08,
                           results_path: str = RESULTS_CSV,
                           ) -> dict:
    """
    Execute the full training pipeline end-to-end.

    Parameters
    ----------
    data_path     : Path to input payroll CSV.
    contamination : Outlier contamination fraction.
    results_path  : Where to save annotated results CSV.

    Returns
    -------
    Dict with keys:
        results_df, models, predictions, scores, ensemble_labels,
        ensemble_scores, feature_importances, feature_names,
        comparison_df, X_scaled, X_df, df_clean
    """
    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    X_scaled, df_clean, scaler, feature_names = preprocess_pipeline(
        data_path, scaler_save_path=SCALER_PATH
    )

    # Rebuild original (unscaled) feature DataFrame for explainability
    df_tmp = load_data(data_path)
    df_tmp = handle_missing_values(df_tmp)
    X_df   = extract_features(df_tmp)

    # ── Step 2: Train all models ──────────────────────────────────────────────
    models, predictions, scores = train_models(
        X_scaled, contamination=contamination, save_models=True
    )

    # ── Step 3: Ensemble ──────────────────────────────────────────────────────
    ensemble_labels = ensemble_predict(predictions, threshold=2)
    ens_scores      = ensemble_score(scores)

    print(f"[train] Ensemble anomalies detected: {ensemble_labels.sum()} "
          f"({ensemble_labels.mean()*100:.1f}%)")

    # ── Step 4: Feature importance ────────────────────────────────────────────
    feature_importances = compute_feature_importance(X_df, ensemble_labels, scaler)

    # ── Step 5: Assemble & save results ──────────────────────────────────────
    results_df = assemble_results(
        df_clean, predictions, scores, ensemble_labels, ens_scores,
        feature_importances, feature_names, save_path=results_path
    )

    # ── Step 6: Comparison table ──────────────────────────────────────────────
    ground_truth = df_clean["is_anomaly"].values if "is_anomaly" in df_clean.columns else None
    comparison_df = build_comparison_table(predictions, ground_truth=ground_truth)

    # Save metadata
    metadata = {
        "contamination":       contamination,
        "n_records":           len(df_clean),
        "n_anomalies":         int(ensemble_labels.sum()),
        "anomaly_pct":         round(float(ensemble_labels.mean()) * 100, 2),
        "feature_names":       feature_names,
        "feature_importances": feature_importances.tolist(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] Metadata saved → {METADATA_PATH}")

    print("\n── Model Comparison ──────────────────────────────────────────────")
    print(comparison_df.to_string(index=False))
    print("───────────────────────────────────────────────────────────────────\n")

    return {
        "results_df":          results_df,
        "models":              models,
        "predictions":         predictions,
        "scores":              scores,
        "ensemble_labels":     ensemble_labels,
        "ensemble_scores":     ens_scores,
        "feature_importances": feature_importances,
        "feature_names":       feature_names,
        "comparison_df":       comparison_df,
        "X_scaled":            X_scaled,
        "X_df":                X_df,
        "df_clean":            df_clean,
        "scaler":              scaler,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output = run_training_pipeline(
        data_path="data/payroll_dataset.csv",
        contamination=0.08,
    )
    df = output["results_df"]
    flagged = df[df["ensemble_anomaly"] == 1]
    print(f"\nFlagged Records Preview ({len(flagged)} total):")
    cols = ["employee_id", "basic_salary", "net_salary", "overtime_hours",
            "bonuses", "tax_deductions", "ensemble_score"]
    available = [c for c in cols if c in flagged.columns]
    print(flagged[available].head(15).to_string(index=False))
