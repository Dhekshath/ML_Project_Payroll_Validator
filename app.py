"""
app.py
======
Streamlit Dashboard for the Payroll Anomaly Detection System.

Tabs
----
  1. 📊 Data Explorer   — Load data, EDA charts, summary statistics
  2. 🚀 Run Detection   — Train all models, pick threshold, see progress
  3. 🔴 Flagged Records — Sortable table with anomaly scores + CSV download
  4. 📈 Visualizations  — PCA, t-SNE, score distribution, feature importance
  5. 🔍 Explainability  — Per-record feature Z-score breakdown
  6. ⚖️ Model Comparison — Side-by-side metrics table + grouped bar chart

Run with:
    streamlit run app.py
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Payroll Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Dark gradient header ── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(79, 195, 247, 0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.main-header h1 {
    color: #4FC3F7;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #90CAF9;
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.85;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
    border: 1px solid rgba(79,195,247,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.8rem;
    color: #90CAF9;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.metric-normal  { color: #4FC3F7; }
.metric-anomaly { color: #FF6B6B; }
.metric-pct     { color: #FFD54F; }
.metric-total   { color: #A5D6A7; }

/* ── Section title ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E0E0E0;
    border-left: 4px solid #4FC3F7;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Anomaly badge ── */
.badge-anomaly {
    background: rgba(255,107,107,0.15);
    color: #FF6B6B;
    border: 1px solid rgba(255,107,107,0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-normal {
    background: rgba(79,195,247,0.10);
    color: #4FC3F7;
    border: 1px solid rgba(79,195,247,0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
    border-right: 1px solid rgba(79,195,247,0.15);
}
[data-testid="stSidebar"] * { color: #E0E0E0 !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #0f0f1e;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #90CAF9;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #4FC3F7 !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background-color: #4FC3F7; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Alert boxes ── */
.info-box {
    background: rgba(79,195,247,0.08);
    border: 1px solid rgba(79,195,247,0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.75rem 0;
    color: #B3E5FC;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key in ["df", "results_df", "X_scaled", "X_df", "feature_names",
            "feature_importances", "predictions", "scores", "ensemble_labels",
            "ensemble_scores", "comparison_df", "scaler", "models_run", "df_clean"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "models_run" not in st.session_state:
    st.session_state["models_run"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Helper: load or generate dataset
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_default_dataset() -> pd.DataFrame:
    """Load from disk or generate a fresh dataset."""
    default_path = "data/payroll_dataset.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    from utils import generate_payroll_dataset
    return generate_payroll_dataset(save_path=default_path)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Parse a user-uploaded CSV file into a DataFrame."""
    content = uploaded_file.read()
    return pd.read_csv(io.BytesIO(content))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem;">🔍</span>
        <h2 style="color:#4FC3F7; font-size:1.15rem; margin:0.25rem 0 0 0; font-weight:700;">
            Payroll Anomaly<br>Detection System
        </h2>
        <p style="color:#78909C; font-size:0.75rem; margin-top:0.3rem;">
            ML-Powered Fraud Detection
        </p>
    </div>
    <hr style="border-color: rgba(79,195,247,0.2); margin: 0.75rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Configuration**")

    contamination = st.slider(
        "Outlier Contamination (%)",
        min_value=1, max_value=30, value=8, step=1,
        help="Expected percentage of anomalies. Adjusts model sensitivity."
    ) / 100.0

    ensemble_threshold = st.selectbox(
        "Ensemble Threshold (# models)",
        options=[1, 2, 3],
        index=1,
        help="Number of models that must agree to flag a record as anomalous."
    )

    viz_mode = st.radio(
        "2D Projection Method",
        options=["PCA", "t-SNE"],
        index=0,
        help="t-SNE is slower but often reveals better cluster separation."
    )

    st.markdown("<hr style='border-color:rgba(79,195,247,0.15);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#546E7A; line-height:1.6;">
        <b style="color:#78909C;">Models Used</b><br>
        • Isolation Forest<br>
        • One-Class SVM<br>
        • Local Outlier Factor<br>
        • Ensemble (Majority Vote)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Payroll Anomaly Detection System</h1>
    <p>
        Unsupervised ML system using Isolation Forest, One-Class SVM &amp; LOF
        to identify suspicious payroll records — powered by ensemble learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Explorer",
    "🚀 Run Detection",
    "🔴 Flagged Records",
    "📈 Visualizations",
    "🔍 Explainability",
    "⚖️ Model Comparison",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📂 Load Payroll Data</p>', unsafe_allow_html=True)

    col_src1, col_src2 = st.columns([1, 1])
    with col_src1:
        uploaded = st.file_uploader(
            "Upload your own CSV file",
            type=["csv"],
            help="Must contain: basic_salary, hra, net_salary, and related columns.",
        )
    with col_src2:
        st.markdown("""
        <div class="info-box">
            <b>Accepted columns:</b> employee_id, basic_salary, hra, other_allowances,
            overtime_hours, bonuses, tax_deductions, other_deductions, net_salary.<br><br>
            <b>No file?</b> Click the button below to use the built-in synthetic dataset.
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        try:
            st.session_state["df"] = load_uploaded_csv(uploaded)
            st.success(f"✅ Uploaded dataset: **{len(st.session_state['df'])}** records")
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")

    if st.button("🔄 Load Built-in Synthetic Dataset", use_container_width=True):
        with st.spinner("Generating realistic payroll dataset..."):
            st.session_state["df"] = get_default_dataset()
        st.success(f"✅ Loaded dataset: **{len(st.session_state['df'])}** records")

    df = st.session_state["df"]

    if df is not None:
        # ── Summary cards
        n_total    = len(df)
        n_cols_num = df.select_dtypes(include=[np.number]).shape[1]
        n_missing  = int(df.isnull().sum().sum())
        n_labeled  = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else "—"

        cm1, cm2, cm3, cm4 = st.columns(4)
        for col, val, label, cls in [
            (cm1, n_total,    "Total Records",     "metric-total"),
            (cm2, n_cols_num, "Numeric Features",  "metric-normal"),
            (cm3, n_missing,  "Missing Values",    "metric-pct"),
            (cm4, n_labeled,  "Known Anomalies",   "metric-anomaly"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-title">🗃️ Raw Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head(200), use_container_width=True, height=300)

        st.markdown('<p class="section-title">📐 Descriptive Statistics</p>', unsafe_allow_html=True)
        from preprocessing import eda_summary_stats
        stats = eda_summary_stats(df)
        st.dataframe(stats, use_container_width=True)

        st.markdown('<p class="section-title">🌡️ Correlation Heatmap</p>', unsafe_allow_html=True)
        from preprocessing import eda_correlation_heatmap, eda_box_plots, eda_histograms, NUMERIC_FEATURES
        st.plotly_chart(eda_correlation_heatmap(df), use_container_width=True)

        st.markdown('<p class="section-title">📦 Feature Box Plots</p>', unsafe_allow_html=True)
        avail_feats = [f for f in NUMERIC_FEATURES if f in df.columns]
        st.plotly_chart(eda_box_plots(df, avail_feats[:6]), use_container_width=True)

        col_hist1, col_hist2 = st.columns(2)
        feat_select = col_hist1.selectbox("Feature distribution", avail_feats, key="hist_feat")
        col_hist2.write("")
        st.plotly_chart(eda_histograms(df, feat_select), use_container_width=True)

    else:
        st.info("⬆️ Upload a CSV or load the built-in dataset to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Run Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🚀 Anomaly Detection Engine</p>', unsafe_allow_html=True)

    if st.session_state["df"] is None:
        st.warning("⚠️ Please load a dataset in the **Data Explorer** tab first.")
    else:
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        col_cfg1.metric("Contamination Rate", f"{contamination:.1%}")
        col_cfg2.metric("Ensemble Threshold", f"{ensemble_threshold} / 3 models")
        col_cfg3.metric("Dataset Size", f"{len(st.session_state['df']):,} records")

        if st.button("▶️  Run Anomaly Detection", use_container_width=True, type="primary"):

            progress   = st.progress(0, text="Initialising pipeline…")
            status_box = st.empty()

            try:
                # Step 1 — Preprocess
                status_box.info("⚙️ Step 1/4 — Preprocessing data…")
                progress.progress(10, text="Preprocessing…")

                # Save the current in-memory df to a temp CSV for the pipeline
                import tempfile, pathlib
                tmp_csv = "data/_app_temp.csv"
                os.makedirs("data", exist_ok=True)
                st.session_state["df"].to_csv(tmp_csv, index=False)

                from preprocessing import preprocess_pipeline, extract_features, load_data, handle_missing_values
                X_scaled, df_clean, scaler, feature_names = preprocess_pipeline(
                    tmp_csv,
                    scaler_save_path=os.path.join("models", "scaler.pkl")
                )
                progress.progress(30, text="Preprocessing complete.")

                # Step 2 — Extract unscaled features
                status_box.info("⚙️ Step 2/4 — Extracting features…")
                df_tmp = load_data(tmp_csv)
                df_tmp = handle_missing_values(df_tmp)
                X_df   = extract_features(df_tmp)
                progress.progress(45, text="Features extracted.")

                # Step 3 — Train models
                status_box.info("⚙️ Step 3/4 — Training Isolation Forest, One-Class SVM, LOF…")
                from train import train_models, ensemble_predict, ensemble_score, compute_feature_importance, build_comparison_table, assemble_results
                models, predictions, scores = train_models(
                    X_scaled, contamination=contamination, save_models=True
                )
                progress.progress(75, text="Models trained.")

                # Step 4 — Ensemble + explainability
                status_box.info("⚙️ Step 4/4 — Computing ensemble & feature importances…")
                ensemble_labels = ensemble_predict(predictions, threshold=ensemble_threshold)
                ens_scores      = ensemble_score(scores)
                feat_imps       = compute_feature_importance(X_df, ensemble_labels, scaler)

                results_df = assemble_results(
                    df_clean, predictions, scores, ensemble_labels, ens_scores,
                    feat_imps, feature_names
                )

                ground_truth = df_clean["is_anomaly"].values if "is_anomaly" in df_clean.columns else None
                comparison_df = build_comparison_table(predictions, ground_truth=ground_truth)

                progress.progress(100, text="Done!")
                status_box.empty()

                # Persist to session state
                st.session_state.update({
                    "results_df":          results_df,
                    "X_scaled":            X_scaled,
                    "X_df":                X_df,
                    "feature_names":       feature_names,
                    "feature_importances": feat_imps,
                    "predictions":         predictions,
                    "scores":              scores,
                    "ensemble_labels":     ensemble_labels,
                    "ensemble_scores":     ens_scores,
                    "comparison_df":       comparison_df,
                    "scaler":              scaler,
                    "models_run":          True,
                    "df_clean":            df_clean,
                })

            except Exception as exc:
                status_box.error(f"❌ Error during detection: {exc}")
                import traceback; st.code(traceback.format_exc())
                progress.empty()

        # ── Results Summary ────────────────────────────────────────────────────
        if st.session_state.get("models_run"):
            labels = st.session_state["ensemble_labels"]
            total  = len(labels)
            n_anom = int(labels.sum())
            n_norm = total - n_anom
            pct    = n_anom / total * 100

            st.markdown("---")
            st.markdown('<p class="section-title">📊 Detection Summary</p>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            for col, val, lbl, cls in [
                (c1, f"{total:,}",     "Total Records",   "metric-total"),
                (c2, f"{n_norm:,}",    "Normal Records",  "metric-normal"),
                (c3, f"{n_anom:,}",    "Anomalies Found", "metric-anomaly"),
                (c4, f"{pct:.1f}%",    "Anomaly Rate",    "metric-pct"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value {cls}">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            st.success("✅ Detection complete! Explore results in the other tabs.")

            # Per-model quick stats
            st.markdown('<p class="section-title">🤖 Per-Model Summary</p>', unsafe_allow_html=True)
            preds = st.session_state["predictions"]
            model_rows = []
            for mname, mpreds in preds.items():
                n_det = (mpreds == -1).sum()
                model_rows.append({
                    "Model": mname,
                    "Anomalies": int(n_det),
                    "Normal": total - int(n_det),
                    "Rate": f"{n_det/total*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Flagged Records
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">🔴 Flagged Anomalous Records</p>', unsafe_allow_html=True)

    if not st.session_state.get("models_run"):
        st.info("⏳ Run anomaly detection first (in the **Run Detection** tab).")
    else:
        results_df = st.session_state["results_df"]
        flagged    = results_df[results_df["ensemble_anomaly"] == 1].copy()

        st.markdown(f"**{len(flagged)} records** flagged as anomalous out of {len(results_df):,} total.")

        # Filters
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        min_score = fc1.slider("Min Ensemble Score", 0.0, 1.0, 0.0, 0.01, key="min_score_filter")
        if "department" in flagged.columns:
            depts = ["All"] + sorted(flagged["department"].dropna().unique().tolist())
            dept_filter = fc2.selectbox("Department", depts, key="dept_filter")
        else:
            dept_filter = "All"

        sort_col = fc3.selectbox(
            "Sort by",
            ["ensemble_score", "basic_salary", "net_salary", "overtime_hours", "bonuses"],
            key="sort_col_filter"
        )

        filtered = flagged[flagged["ensemble_score"] >= min_score]
        if dept_filter != "All" and "department" in filtered.columns:
            filtered = filtered[filtered["department"] == dept_filter]

        if sort_col in filtered.columns:
            filtered = filtered.sort_values(sort_col, ascending=False)

        # Display columns
        display_cols = [c for c in [
            "employee_id", "department", "basic_salary", "hra", "other_allowances",
            "overtime_hours", "bonuses", "tax_deductions", "net_salary",
            "ensemble_score",
            "isolation_forest_pred", "oneclasssvm_pred", "localoutlierfactor_pred",
            "anomaly_type",
        ] if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        # Download button
        csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Flagged Records (CSV)",
            data=csv_bytes,
            file_name="anomaly_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Score heatmap mini-table
        st.markdown('<p class="section-title">🌡️ Anomaly Score Heatmap (Top 30)</p>',
                    unsafe_allow_html=True)
        score_cols = [c for c in ["isolation_forest_score", "oneclasssvm_score",
                                   "localoutlierfactor_score", "ensemble_score"]
                      if c in filtered.columns]
        if score_cols and "employee_id" in filtered.columns:
            top30 = filtered.head(30)[["employee_id"] + score_cols].set_index("employee_id")
            fig_heat = px.imshow(
                top30.T,
                color_continuous_scale="YlOrRd",
                title="Anomaly Scores per Record (Top 30)",
                template="plotly_dark",
                aspect="auto",
                text_auto=".2f",
            )
            fig_heat.update_layout(margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">📈 Anomaly Visualizations</p>', unsafe_allow_html=True)

    if not st.session_state.get("models_run"):
        st.info("⏳ Run anomaly detection first.")
    else:
        X_scaled         = st.session_state["X_scaled"]
        ensemble_labels  = st.session_state["ensemble_labels"]
        ensemble_scores  = st.session_state["ensemble_scores"]
        feature_names    = st.session_state["feature_names"]
        feature_imps     = st.session_state["feature_importances"]

        from utils import (plot_pca_scatter, plot_tsne_scatter,
                           plot_score_distribution, plot_feature_importance)

        # ── 2D Projection ─────────────────────────────────────────────────────
        st.markdown('<p class="section-title">🗺️ 2D Projection Plot</p>', unsafe_allow_html=True)
        with st.spinner(f"Computing {viz_mode} projection (may take a moment)…"):
            if viz_mode == "PCA":
                fig_2d = plot_pca_scatter(X_scaled, ensemble_labels, ensemble_scores)
            else:
                fig_2d = plot_tsne_scatter(X_scaled, ensemble_labels)
        st.plotly_chart(fig_2d, use_container_width=True)

        # ── Score Distribution ────────────────────────────────────────────────
        st.markdown('<p class="section-title">📊 Anomaly Score Distribution</p>',
                    unsafe_allow_html=True)
        threshold_line = sorted(ensemble_scores)[int(len(ensemble_scores) * (1 - contamination))]
        st.plotly_chart(
            plot_score_distribution(ensemble_scores, ensemble_labels, threshold=threshold_line),
            use_container_width=True
        )

        # ── Feature Importance ────────────────────────────────────────────────
        st.markdown('<p class="section-title">🏆 Feature Importance</p>', unsafe_allow_html=True)
        if feature_imps is not None and len(feature_imps) == len(feature_names):
            st.plotly_chart(
                plot_feature_importance(feature_names, feature_imps),
                use_container_width=True
            )

        # ── Scatter: salary vs overtime ───────────────────────────────────────
        st.markdown('<p class="section-title">💰 Salary vs Overtime Scatter</p>',
                    unsafe_allow_html=True)
        results_df = st.session_state["results_df"]
        if "basic_salary" in results_df.columns and "overtime_hours" in results_df.columns:
            fig_scatter = px.scatter(
                results_df,
                x="basic_salary",
                y="overtime_hours",
                color=results_df["ensemble_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                color_discrete_map={"Normal": "#4FC3F7", "Anomaly": "#FF6B6B"},
                size="ensemble_score",
                size_max=18,
                opacity=0.7,
                hover_data=["employee_id", "net_salary", "bonuses"] if "employee_id" in results_df.columns else None,
                title="Basic Salary vs Overtime Hours (sized by Anomaly Score)",
                template="plotly_dark",
                labels={"basic_salary": "Basic Salary (₹)", "overtime_hours": "Overtime Hours"},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explainability
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🔍 Why Is This Record Anomalous?</p>',
                unsafe_allow_html=True)

    if not st.session_state.get("models_run"):
        st.info("⏳ Run anomaly detection first.")
    else:
        results_df = st.session_state["results_df"]
        X_df       = st.session_state["X_df"]
        labels     = st.session_state["ensemble_labels"]

        flagged_df = results_df[results_df["ensemble_anomaly"] == 1]

        if len(flagged_df) == 0:
            st.warning("No anomalies detected at the current threshold.")
        else:
            # Record selector
            id_col    = "employee_id" if "employee_id" in flagged_df.columns else flagged_df.index.astype(str)
            id_list   = flagged_df["employee_id"].tolist() if "employee_id" in flagged_df.columns \
                        else [str(i) for i in flagged_df.index]

            selected_id = st.selectbox(
                "Select anomalous Employee ID to explain",
                id_list,
                key="explain_select"
            )

            if selected_id:
                row_idx = flagged_df[flagged_df["employee_id"] == selected_id].index[0] \
                          if "employee_id" in flagged_df.columns else int(selected_id)

                from train import explain_record
                record_features = X_df.iloc[row_idx] if row_idx < len(X_df) else X_df.iloc[0]
                explanation     = explain_record(record_features, X_df, labels)

                # ── Record details card ───────────────────────────────────────
                st.markdown(f"**Record:** `{selected_id}`  |  "
                            f"**Ensemble Score:** `{results_df.loc[row_idx, 'ensemble_score']:.4f}`")

                rec_cols = [c for c in ["basic_salary", "hra", "overtime_hours",
                                         "bonuses", "tax_deductions", "net_salary"]
                             if c in results_df.columns]
                if rec_cols:
                    rc_vals = results_df.loc[row_idx, rec_cols]
                    rc_df   = pd.DataFrame({"Feature": rc_vals.index, "Value": rc_vals.values})
                    col_d1, col_d2 = st.columns([1, 2])
                    col_d1.dataframe(rc_df, use_container_width=True, hide_index=True)

                # ── Z-Score table ─────────────────────────────────────────────
                st.markdown('<p class="section-title">📐 Feature Z-Score Analysis</p>',
                            unsafe_allow_html=True)
                st.dataframe(
                    explanation.style.background_gradient(
                        subset=["Z-Score"], cmap="RdYlGn_r"
                    ).format({"Z-Score": "{:.3f}", "Normal Mean": "{:,.2f}",
                              "Record Value": "{:,.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── Waterfall-style bar chart ─────────────────────────────────
                st.markdown('<p class="section-title">📊 Feature Deviation Chart</p>',
                            unsafe_allow_html=True)
                exp_top = explanation.head(10).copy()
                colors  = ["#FF6B6B" if f else "#4FC3F7" for f in exp_top["Flagged"]]
                fig_exp = go.Figure(go.Bar(
                    x=exp_top["Z-Score"],
                    y=exp_top["Feature"],
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="%{y}: Z=%{x:.3f}<extra></extra>",
                ))
                fig_exp.update_layout(
                    title=f"Feature Z-Scores for {selected_id}  (|Z| > 3 = suspicious 🚨)",
                    xaxis_title="Z-Score (standard deviations from normal mean)",
                    template="plotly_dark",
                    shapes=[{
                        "type": "rect", "xref": "x", "yref": "paper",
                        "x0": -3, "x1": 3, "y0": 0, "y1": 1,
                        "fillcolor": "rgba(79,195,247,0.07)",
                        "line_width": 0,
                    }],
                    annotations=[
                        {"x": 3, "y": 1.05, "xref": "x", "yref": "paper",
                         "text": "Normal zone (±3σ)", "showarrow": False,
                         "font": {"color": "#4FC3F7", "size": 11}},
                    ],
                    margin=dict(l=40, r=40, t=70, b=40),
                )
                st.plotly_chart(fig_exp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">⚖️ Model Comparison</p>', unsafe_allow_html=True)

    if not st.session_state.get("models_run"):
        st.info("⏳ Run anomaly detection first.")
    else:
        comparison_df = st.session_state["comparison_df"]
        predictions   = st.session_state["predictions"]
        labels        = st.session_state["ensemble_labels"]

        # Comparison table
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        from utils import plot_model_comparison

        # Bar chart
        metric_cols = [c for c in comparison_df.columns
                       if c not in ("Model",) and comparison_df[c].dtype != object]
        if metric_cols:
            chart_df = comparison_df[comparison_df["Anomalies Detected"] != "—"].copy()
            chart_df["Anomalies Detected"] = chart_df["Anomalies Detected"].astype(int)
            chart_df["Detection Rate (%)"] = chart_df["Detection Rate (%)"].astype(float)
            fig_cmp = plot_model_comparison(chart_df[["Model", "Anomalies Detected", "Detection Rate (%)"]])
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Agreement heatmap
        st.markdown('<p class="section-title">🤝 Model Agreement Heatmap</p>',
                    unsafe_allow_html=True)
        model_names = list(predictions.keys())
        n_models    = len(model_names)
        agree_matrix = np.zeros((n_models, n_models))

        for i, mi in enumerate(model_names):
            for j, mj in enumerate(model_names):
                pi = (predictions[mi] == -1).astype(int)
                pj = (predictions[mj] == -1).astype(int)
                agree_matrix[i, j] = (pi == pj).mean()

        fig_agree = px.imshow(
            agree_matrix,
            x=model_names, y=model_names,
            color_continuous_scale="Blues",
            zmin=0, zmax=1,
            text_auto=".2%",
            title="Pairwise Model Agreement Rate",
            template="plotly_dark",
        )
        st.plotly_chart(fig_agree, use_container_width=True)

        # Venn-style per-record agreement
        st.markdown('<p class="section-title">🔢 Detection Agreement Breakdown</p>',
                    unsafe_allow_html=True)
        vote_arr = np.stack([(p == -1).astype(int) for p in predictions.values()], axis=1)
        vote_sum = vote_arr.sum(axis=1)
        counts   = pd.Series(vote_sum).value_counts().sort_index()
        vote_labels = {0: "No model flagged", 1: "1 model", 2: "2 models", 3: "All 3 models"}
        vote_df = pd.DataFrame({
            "Agreement": [vote_labels.get(k, f"{k} models") for k in counts.index],
            "Count": counts.values,
            "Percentage": (counts.values / len(vote_arr) * 100).round(2),
        })
        col_venn1, col_venn2 = st.columns([1, 2])
        col_venn1.dataframe(vote_df, use_container_width=True, hide_index=True)
        fig_donut = px.pie(
            vote_df, values="Count", names="Agreement",
            title="Detection Agreement Distribution",
            template="plotly_dark",
            hole=0.45,
            color_discrete_sequence=["#2E4057", "#4FC3F7", "#FFD54F", "#FF6B6B"],
        )
        col_venn2.plotly_chart(fig_donut, use_container_width=True)
