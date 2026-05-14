# 🔍 Payroll Anomaly Detection System

**Machine Learning-powered system for detecting suspicious payroll records using unsupervised learning.**

---

## 📌 Problem Statement

Payroll fraud costs organizations billions every year. Traditional rule-based validators (e.g., "flag if salary > X") are brittle, easy to circumvent, and require constant manual updating. 

This system takes a fundamentally different approach: **unsupervised machine learning** learns the "normal" distribution of payroll data and flags records that deviate from it — without needing labeled data or handcrafted rules.

---

## 🎯 Objective

Build a production-quality anomaly detection pipeline that:
1. Detects unusual payroll records using ML (not rules)
2. Quantifies *how anomalous* each record is (anomaly score)
3. Explains *why* a record is flagged (feature-level Z-scores)
4. Compares multiple ML models to find the best approach
5. Provides an interactive UI for upload, detection, and reporting

---

## 🏗️ Architecture

```
Payroll Validator/
├── data/
│   ├── payroll_dataset.csv      # 1220-record synthetic dataset
│   └── anomaly_results.csv      # Output: all records with anomaly labels + scores
├── models/
│   ├── isolation_forest.pkl     # Trained Isolation Forest
│   ├── oneclass_svm.pkl         # Trained One-Class SVM
│   ├── scaler.pkl               # Fitted StandardScaler
│   └── metadata.json            # Training run metadata
├── app.py                       # Streamlit UI (6 tabs)
├── preprocessing.py             # Data loading, cleaning, scaling, EDA
├── train.py                     # All model logic, ensemble, explainability
├── utils.py                     # Dataset generator, plotting helpers
├── requirements.txt
└── README.md
```

---

## 🤖 Approach

### Dataset
- **1,220 records** (1,100 normal + 120 synthetic anomalies)
- Features: `basic_salary`, `hra`, `other_allowances`, `overtime_hours`, `bonuses`, `tax_deductions`, `other_deductions`, `net_salary`, `years_of_service`, `performance_score`
- Derived features: `net_to_gross_ratio`, `tax_rate`, `bonus_rate`, `high_overtime_flag`

### Anomaly Types Injected
| Type | Description |
|------|-------------|
| `salary_spike` | Salary is 4–10x the expected range |
| `negative_deduction` | Negative tax/deduction (fraudulent write-back) |
| `impossible_overtime` | 200–400 overtime hours in a single month |
| `bonus_abuse` | Bonus is 3–7x the base salary |
| `net_salary_inconsistency` | Net salary doesn't match the arithmetic |
| `ghost_employee` | Zero salary but receives large allowances/bonuses |
| `duplicate_manipulation` | Cloned record with inflated bonus, tiny tax |

### Preprocessing Pipeline
1. Load CSV → validate required columns
2. Median imputation for missing values (~2% introduced for realism)
3. Feature engineering (derived ratios, flags)
4. Clip extreme outliers at 0.5th / 99.5th percentiles
5. StandardScaler normalization

### Models

#### 1. Isolation Forest *(Primary)*
- Anomaly detection via random feature partitioning
- Isolates anomalies faster (fewer splits needed)
- Contamination parameter tunable via UI slider

#### 2. One-Class SVM *(Comparison)*
- Learns a hypersphere boundary around normal data
- Points outside the boundary = anomalies
- Uses RBF kernel; `nu` = contamination rate

#### 3. Local Outlier Factor *(Comparison)*
- Compares local density of each point to its neighbors
- Excellent at detecting context-dependent anomalies
- k=20 neighbors; transductive (fit+predict together)

#### 4. Ensemble (Majority Vote)
- Flags a record if **≥ 2 of 3 models** agree
- More conservative, reduces false positives
- Threshold configurable in the UI (1, 2, or 3 models)

---

## 📊 Results

| Model | Anomalies Detected | Detection Rate | Precision | Recall | F1-Score |
|-------|-------------------|----------------|-----------|--------|----------|
| **Isolation Forest** | 98 | 8.03% | **0.8980** | **0.7333** | **0.8073** |
| One-Class SVM | 99 | 8.11% | 0.8182 | 0.6750 | 0.7397 |
| Local Outlier Factor | 98 | 8.03% | 0.7143 | 0.5833 | 0.6422 |
| **Ensemble (Majority Vote)** | **99** | **8.11%** | — | — | — |

> **Isolation Forest** delivers the highest precision (0.898) and F1-score (0.807) and is selected as the primary model.

---

## 🔍 Explainability

For every flagged record, the system provides:
- **Z-score analysis**: How many standard deviations each feature is from the normal population mean
- **Feature deviation chart**: Visual bar chart showing the top deviating features
- **Anomaly score**: Continuous score from 0.0 (very normal) to 1.0 (very anomalous)

Features with |Z| > 3 are highlighted as suspicious.

---

## 🖥️ Streamlit UI

The dashboard has **6 interactive tabs**:

| Tab | Description |
|-----|-------------|
| 📊 **Data Explorer** | Upload CSV or load built-in data. EDA: correlation heatmap, box plots, histograms |
| 🚀 **Run Detection** | Configure contamination %, run all 3 models + ensemble, view detection summary |
| 🔴 **Flagged Records** | Filter, sort, and download anomalous records with all model scores |
| 📈 **Visualizations** | PCA/t-SNE 2D scatter, anomaly score distribution, salary vs overtime scatter |
| 🔍 **Explainability** | Select any flagged record; see Z-score table + feature deviation waterfall chart |
| ⚖️ **Model Comparison** | Side-by-side metrics table, pairwise agreement heatmap, detection voting pie chart |

---

## 🚀 How to Run

### 1. Clone / navigate to the project directory
```bash
cd "e:\ML project\Payroll Validator"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset (first time only)
```bash
python utils.py
```

### 4. Pre-train the models (optional — app auto-trains on button click)
```bash
python train.py
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📥 Input CSV Format

If uploading your own CSV, include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `employee_id` | string | Unique employee identifier |
| `basic_salary` | float | Base monthly salary |
| `hra` | float | House rent allowance |
| `other_allowances` | float | Additional allowances |
| `overtime_hours` | float | Overtime hours logged |
| `bonuses` | float | Bonus payment |
| `tax_deductions` | float | Tax withheld |
| `other_deductions` | float | Other deductions |
| `net_salary` | float | Final take-home pay |

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Contamination | 8% | Expected anomaly rate (affects all 3 models) |
| Ensemble Threshold | 2 | # models that must agree to flag a record |
| Projection | PCA | Dimensionality reduction for visualization |

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.20.0
joblib>=1.3.0
```

---

## 📝 Notes

- The system uses **unsupervised learning only** — no labeled training data is required
- Models are saved to `models/` and reloaded automatically between runs
- Results are saved to `data/anomaly_results.csv` for audit trails
- All visualizations are interactive (Plotly) and embeddable in reports

---

*Built with Python · scikit-learn · Streamlit · Plotly*
