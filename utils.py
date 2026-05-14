"""
utils.py
========
Utility functions for the Payroll Anomaly Detection System.

Responsibilities:
  - Generate a realistic synthetic payroll dataset with injected anomalies
  - Provide reusable plotting helpers (PCA scatter, t-SNE, score distribution,
    feature importance bar chart)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Seed for reproducibility
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_payroll_dataset(n_normal: int = 1100, n_anomalies: int = 120,
                              save_path: str = "data/payroll_dataset.csv") -> pd.DataFrame:
    """
    Generate a realistic payroll dataset with injected anomalies and save to CSV.

    Parameters
    ----------
    n_normal    : Number of normal (clean) payroll records.
    n_anomalies : Number of injected anomalous records.
    save_path   : File path to save the generated CSV dataset.

    Returns
    -------
    pd.DataFrame containing the full dataset (normal + anomalous).
    """
    # ── Normal Records ────────────────────────────────────────────────────────
    employee_ids = [f"EMP{str(i).zfill(5)}" for i in range(1, n_normal + n_anomalies + 1)]

    # Department-based salary tiers to make the data realistic
    dept_salaries = {
        "Engineering": (55000, 12000),
        "Sales":       (42000, 9000),
        "HR":          (38000, 7000),
        "Finance":     (50000, 11000),
        "Operations":  (35000, 8000),
        "Management":  (80000, 18000),
    }
    dept_names = list(dept_salaries.keys())
    dept_weights = [0.25, 0.20, 0.10, 0.15, 0.20, 0.10]

    departments = np.random.choice(dept_names, size=n_normal, p=dept_weights)

    basic_salary = np.array([
        max(20000, np.random.normal(dept_salaries[d][0], dept_salaries[d][1]))
        for d in departments
    ])

    hra = np.round(basic_salary * np.random.uniform(0.30, 0.50, size=n_normal), 2)
    other_allowances = np.round(basic_salary * np.random.uniform(0.10, 0.25, size=n_normal), 2)

    # Overtime: 0–60 h/month; most employees have low overtime
    overtime_hours = np.round(
        np.clip(np.random.exponential(scale=8, size=n_normal), 0, 60), 1
    )
    overtime_rate  = (basic_salary / 160) * 1.5          # 1.5× hourly rate
    overtime_pay   = np.round(overtime_hours * overtime_rate, 2)

    bonuses = np.round(
        np.where(np.random.rand(n_normal) < 0.35,         # 35% receive a bonus
                 basic_salary * np.random.uniform(0.05, 0.20, size=n_normal),
                 0), 2
    )

    tax_deductions = np.round(
        basic_salary * np.random.uniform(0.10, 0.30, size=n_normal), 2
    )
    other_deductions = np.round(
        basic_salary * np.random.uniform(0.02, 0.08, size=n_normal), 2
    )

    gross_salary = basic_salary + hra + other_allowances + overtime_pay + bonuses
    net_salary   = np.round(gross_salary - tax_deductions - other_deductions, 2)

    years_of_service = np.random.randint(1, 30, size=n_normal)
    performance_score = np.round(np.random.uniform(40, 100, size=n_normal), 1)

    normal_df = pd.DataFrame({
        "employee_id":      employee_ids[:n_normal],
        "department":       departments,
        "years_of_service": years_of_service,
        "performance_score": performance_score,
        "basic_salary":     np.round(basic_salary, 2),
        "hra":              hra,
        "other_allowances": other_allowances,
        "overtime_hours":   overtime_hours,
        "bonuses":          bonuses,
        "tax_deductions":   tax_deductions,
        "other_deductions": other_deductions,
        "net_salary":       net_salary,
        "is_anomaly":       0,
        "anomaly_type":     "Normal",
    })

    # ── Anomalous Records ─────────────────────────────────────────────────────
    anomaly_types = [
        "salary_spike",
        "negative_deduction",
        "impossible_overtime",
        "bonus_abuse",
        "net_salary_inconsistency",
        "ghost_employee",
        "duplicate_manipulation",
    ]
    anomaly_weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05]

    chosen_types = np.random.choice(anomaly_types, size=n_anomalies, p=anomaly_weights)
    anomaly_records = []

    for i, atype in enumerate(chosen_types):
        emp_id = employee_ids[n_normal + i]
        dept   = np.random.choice(dept_names)
        base   = np.random.normal(dept_salaries[dept][0], dept_salaries[dept][1])
        base   = max(20000, base)
        yrs    = np.random.randint(1, 30)
        perf   = round(np.random.uniform(40, 100), 1)

        if atype == "salary_spike":
            # Unrealistically high salary
            basic = round(base * np.random.uniform(4, 10), 2)
            hra_v = round(basic * 0.40, 2)
            other = round(basic * 0.15, 2)
            ovt_h = round(np.random.uniform(0, 30), 1)
            ovt_p = round(ovt_h * (basic / 160) * 1.5, 2)
            bon   = round(basic * 0.15, 2)
            tax   = round(basic * 0.20, 2)
            oth_d = round(basic * 0.05, 2)
            net   = round(basic + hra_v + other + ovt_p + bon - tax - oth_d, 2)

        elif atype == "negative_deduction":
            # Negative tax or other deductions (fraudulent)
            basic = round(base, 2)
            hra_v = round(basic * 0.40, 2)
            other = round(basic * 0.15, 2)
            ovt_h = round(np.random.uniform(0, 20), 1)
            ovt_p = round(ovt_h * (basic / 160) * 1.5, 2)
            bon   = round(basic * 0.10, 2)
            tax   = round(-abs(np.random.uniform(500, 5000)), 2)   # NEGATIVE!
            oth_d = round(-abs(np.random.uniform(100, 1000)), 2)   # NEGATIVE!
            net   = round(basic + hra_v + other + ovt_p + bon - tax - oth_d, 2)

        elif atype == "impossible_overtime":
            # More overtime hours than physically possible in a month
            basic = round(base, 2)
            hra_v = round(basic * 0.40, 2)
            other = round(basic * 0.15, 2)
            ovt_h = round(np.random.uniform(200, 400), 1)   # impossible (>168 h in a month)
            ovt_p = round(ovt_h * (basic / 160) * 1.5, 2)
            bon   = round(basic * 0.10, 2)
            tax   = round(basic * 0.20, 2)
            oth_d = round(basic * 0.05, 2)
            net   = round(basic + hra_v + other + ovt_p + bon - tax - oth_d, 2)

        elif atype == "bonus_abuse":
            # Bonus exceeds salary entirely
            basic = round(base, 2)
            hra_v = round(basic * 0.40, 2)
            other = round(basic * 0.15, 2)
            ovt_h = round(np.random.uniform(0, 20), 1)
            ovt_p = round(ovt_h * (basic / 160) * 1.5, 2)
            bon   = round(basic * np.random.uniform(3, 7), 2)   # massive bonus
            tax   = round(basic * 0.20, 2)
            oth_d = round(basic * 0.05, 2)
            net   = round(basic + hra_v + other + ovt_p + bon - tax - oth_d, 2)

        elif atype == "net_salary_inconsistency":
            # Net salary doesn't match the expected calculation
            basic = round(base, 2)
            hra_v = round(basic * 0.40, 2)
            other = round(basic * 0.15, 2)
            ovt_h = round(np.random.uniform(0, 30), 1)
            ovt_p = round(ovt_h * (basic / 160) * 1.5, 2)
            bon   = round(basic * 0.10, 2)
            tax   = round(basic * 0.20, 2)
            oth_d = round(basic * 0.05, 2)
            # Net is wildly different from expected
            net   = round((basic + hra_v + other + ovt_p + bon - tax - oth_d)
                          * np.random.choice([3.5, 0.1, -0.5]), 2)

        elif atype == "ghost_employee":
            # Zero salary but receives allowances/bonuses
            basic = 0.0
            hra_v = round(np.random.uniform(5000, 20000), 2)
            other = round(np.random.uniform(2000, 10000), 2)
            ovt_h = 0.0
            ovt_p = 0.0
            bon   = round(np.random.uniform(10000, 50000), 2)
            tax   = 0.0
            oth_d = 0.0
            net   = round(hra_v + other + bon, 2)

        else:  # duplicate_manipulation
            # Clone an existing record with slight salary manipulation
            src = normal_df.iloc[np.random.randint(0, len(normal_df))]
            basic = round(float(src["basic_salary"]) * np.random.uniform(0.95, 1.05), 2)
            hra_v = round(basic * 0.40, 2)
            other = round(basic * 0.15, 2)
            ovt_h = float(src["overtime_hours"])
            ovt_p = round(ovt_h * (basic / 160) * 1.5, 2)
            bon   = round(basic * 0.50, 2)   # inflated bonus
            tax   = round(basic * 0.05, 2)   # suspiciously low tax
            oth_d = round(basic * 0.02, 2)
            net   = round(basic + hra_v + other + ovt_p + bon - tax - oth_d, 2)
            yrs   = int(src["years_of_service"])
            perf  = float(src["performance_score"])

        anomaly_records.append({
            "employee_id":      emp_id,
            "department":       dept,
            "years_of_service": yrs,
            "performance_score": perf,
            "basic_salary":     basic,
            "hra":              hra_v,
            "other_allowances": other,
            "overtime_hours":   ovt_h,
            "bonuses":          bon,
            "tax_deductions":   tax,
            "other_deductions": oth_d,
            "net_salary":       net,
            "is_anomaly":       1,
            "anomaly_type":     atype,
        })

    anomaly_df = pd.DataFrame(anomaly_records)

    # ── Combine and Shuffle ───────────────────────────────────────────────────
    full_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Introduce ~2% missing values in non-critical columns for realism
    for col in ["performance_score", "bonuses", "other_allowances"]:
        mask = np.random.rand(len(full_df)) < 0.02
        full_df.loc[mask, col] = np.nan

    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_df.to_csv(save_path, index=False)
    print(f"[utils] Dataset saved -> {save_path}  "
          f"({len(normal_df)} normal + {len(anomaly_df)} anomalies = {len(full_df)} total)")
    return full_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_numeric_features() -> list:
    """Return the list of numeric feature columns used for ML training."""
    return [
        "basic_salary", "hra", "other_allowances",
        "overtime_hours", "bonuses", "tax_deductions",
        "other_deductions", "net_salary",
        "years_of_service", "performance_score",
    ]


def plot_pca_scatter(X_scaled: np.ndarray, labels: np.ndarray,
                     scores: np.ndarray | None = None,
                     title: str = "PCA — Anomaly Detection") -> go.Figure:
    """
    Reduce features to 2D using PCA and render an interactive Plotly scatter.

    Parameters
    ----------
    X_scaled : Scaled feature matrix (n_samples × n_features).
    labels   : Binary array — 1 = anomaly, 0 = normal.
    scores   : Optional anomaly scores for marker sizing.
    title    : Chart title.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    variance_explained = pca.explained_variance_ratio_ * 100
    color_map = {0: "steelblue", 1: "crimson"}
    colors = [color_map[l] for l in labels]
    label_names = ["Normal" if l == 0 else "Anomaly" for l in labels]

    size = None
    if scores is not None:
        # Map scores to marker size (3–18)
        s_min, s_max = scores.min(), scores.max()
        size_arr = 3 + 15 * (scores - s_min) / (s_max - s_min + 1e-9)
        size = size_arr.tolist()

    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        color=label_names,
        size=size if size else None,
        color_discrete_map={"Normal": "steelblue", "Anomaly": "crimson"},
        labels={"x": f"PC1 ({variance_explained[0]:.1f}% var)",
                "y": f"PC2 ({variance_explained[1]:.1f}% var)",
                "color": "Record Type"},
        title=title,
        opacity=0.75,
        template="plotly_dark",
    )
    fig.update_layout(
        title_font_size=16,
        legend_title_text="Record Type",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_tsne_scatter(X_scaled: np.ndarray, labels: np.ndarray,
                      title: str = "t-SNE — Anomaly Detection") -> go.Figure:
    """
    Reduce features to 2D using t-SNE and render an interactive Plotly scatter.
    """
    perplexity = min(30, len(X_scaled) // 4)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE,
                n_iter=800, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(X_scaled)

    label_names = ["Normal" if l == 0 else "Anomaly" for l in labels]
    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        color=label_names,
        color_discrete_map={"Normal": "steelblue", "Anomaly": "crimson"},
        labels={"x": "t-SNE Dim 1", "y": "t-SNE Dim 2", "color": "Record Type"},
        title=title,
        opacity=0.75,
        template="plotly_dark",
    )
    fig.update_layout(
        title_font_size=16,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_score_distribution(scores: np.ndarray, labels: np.ndarray,
                             threshold: float | None = None) -> go.Figure:
    """
    Plot anomaly score distribution overlaid for normal and anomalous records.
    """
    score_df = pd.DataFrame({"score": scores, "type": ["Anomaly" if l == 1 else "Normal"
                                                         for l in labels]})
    fig = px.histogram(
        score_df, x="score", color="type",
        nbins=60, barmode="overlay", opacity=0.7,
        color_discrete_map={"Normal": "steelblue", "Anomaly": "crimson"},
        labels={"score": "Anomaly Score", "count": "Count"},
        title="Anomaly Score Distribution",
        template="plotly_dark",
    )
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="gold",
                      annotation_text=f"Threshold ({threshold:.3f})",
                      annotation_position="top right")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_feature_importance(feature_names: list, importances: np.ndarray,
                             title: str = "Mean Feature Importance (Anomalous Records)") -> go.Figure:
    """
    Horizontal bar chart showing which features drive anomaly detection most.
    """
    sorted_idx = np.argsort(importances)
    sorted_names  = [feature_names[i] for i in sorted_idx]
    sorted_values = importances[sorted_idx]

    colors = [
        f"rgba({int(255 * v / max(sorted_values))}, "
        f"{int(80 * (1 - v / max(sorted_values)))}, "
        f"{int(200 * (1 - v / max(sorted_values)))}, 0.85)"
        for v in sorted_values
    ]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_names,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Mean |Deviation| from Normal",
        yaxis_title="Feature",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart for side-by-side model comparison metrics.
    """
    metrics = [c for c in comparison_df.columns if c != "Model"]
    fig = go.Figure()
    palette = ["#4FC3F7", "#FF8A65", "#AED581"]

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df["Model"],
            y=comparison_df[metric],
            marker_color=palette[i % len(palette)],
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        xaxis_title="Model",
        yaxis_title="Value",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Metric",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — generate data when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_payroll_dataset()
    print(df.head(10).to_string())
    print(f"\nAnomaly breakdown:\n{df['anomaly_type'].value_counts()}")
