# ============================================================
# app.py — Credit Risk Scoring & Multi-Model Benchmarking
# Senior-Engineer Portfolio Showcase & Live Underwriting App
# Run via: run_app.bat  or  streamlit run app.py
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

st.set_page_config(
    page_title="Credit Risk Scoring — Portfolio Showcase",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# Pipeline Transformer (required for joblib model deserialization)
# ─────────────────────────────────────────────────────────────
def add_financial_ratios(df_input: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 Indian banking domain-specific financial ratios."""
    df_out = df_input.copy()
    inc = df_out["monthly_income_inr"].replace(0, np.nan)
    tenure = df_out["loan_tenure_months"].replace(0, np.nan)

    df_out["emi_to_income_ratio"] = (df_out["existing_emi_monthly_inr"] / inc).fillna(0).round(4)
    df_out["loan_to_annual_income_ratio"] = (df_out["loan_amount_requested_inr"] / (inc * 12)).fillna(0).round(4)
    df_out["estimated_monthly_installment"] = (df_out["loan_amount_requested_inr"] / tenure).fillna(0).round(2)
    df_out["installment_to_income_ratio"] = (df_out["estimated_monthly_installment"] / inc).fillna(0).round(4)
    total_emi = df_out["existing_emi_monthly_inr"].fillna(0) + df_out["estimated_monthly_installment"].fillna(0)
    df_out["total_obligations_ratio"] = (total_emi / inc).fillna(0).round(4)
    deficit = (900 - df_out["cibil_score"].fillna(750)) / 100
    df_out["credit_hunger_index"] = (df_out["num_credit_inquiries_last_6m"].fillna(0) * deficit).round(4)
    df_out["cibil_band_code"] = df_out["cibil_score"].apply(
        lambda s: 0 if pd.isna(s) or s < 650 else 1 if s < 700 else 2 if s < 750 else 3
    )
    return df_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for Indian banking ratio engineering."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a DataFrame")
        return add_financial_ratios(X)


# ─────────────────────────────────────────────────────────────
# Cached Data & Model Loaders
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(filename: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_models() -> dict:
    models = {}
    if not os.path.exists(MODELS_DIR):
        return models
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith(".joblib"):
            label = fname.replace(".joblib", "").replace("_", " ")
            models[label] = joblib.load(os.path.join(MODELS_DIR, fname))
    return models


def show_result_image(filename: str, caption: str, alt: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.info(
            f"Chart `{filename}` not found. Run `notebooks/01_EDA_and_Data_Cleaning.ipynb` "
            "and `notebooks/02_Modelling_Benchmarking_SHAP.ipynb` to generate it.",
            title="Artifact missing",
        )


def classify_risk_tier(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "LOW RISK", "green"
    if prob < 0.60:
        return "MEDIUM RISK", "orange"
    return "HIGH RISK", "red"


# ─────────────────────────────────────────────────────────────
# PAGE 1 — Executive Summary & Architecture
# ─────────────────────────────────────────────────────────────
def page_overview() -> None:
    st.title("Credit Risk Scoring & Underwriting Engine")
    st.caption(
        "End-to-End Machine Learning System · India Personal Loan Default Risk (25,000 Applicants) · "
        "Leakage-Free Pipelines, Cost-Sensitive Thresholding & SHAP Explainability"
    )

    with st.grid(4):
        st.metric("Applicants Audited", "25,000", "6.14% default rate (1:15.3)", border=True)
        st.metric("Best PR-AUC", "0.3832", "6.2x random baseline (0.0614)", border=True)
        st.metric("Best ROC-AUC", "0.8728", "Logistic Regression (HistGB 0.8719)", border=True)
        st.metric("Loss Reduction", "-22.6%", "Cost 1,314 -> 1,017 at t=0.694", border=True)

    st.space("small")

    col_left, col_right = st.columns([1.15, 1.0], gap="medium")

    with col_left:
        with st.container(border=True):
            st.subheader("Business Problem & Asymmetric Risk")
            st.markdown(
                """
In personal lending (banks, NBFCs, FinTechs), classification errors have **asymmetric financial consequences**:

| Decision Error | Real-World Impact | Cost Weight |
|:---|:---|:---|
| **False Negative** (approve a defaulter) | Direct principal write-off + collections & recovery cost | **5x** |
| **False Positive** (reject a creditworthy borrower) | Foregone interest margin + customer churn | **1x** |

With only **6.14% defaults**, a naive model predicting *"repay"* for every borrower hits **93.86% accuracy** while catching **0% of defaults**. This system optimizes **PR-AUC, Recall, and Expected Business Cost** instead of accuracy.
                """
            )

    with col_right:
        with st.container(border=True):
            st.subheader("Engineering Rigor & Leakage Prevention")
            st.markdown(
                """
- **Target Leakage Eliminated:** Detected and dropped `default_risk_score` (and `applicant_id`) during the initial audit before any modeling.
- **Strict Split-Before-Transform:** Stratified 80/20 split (`20,000` train / `5,000` test) executed before winsorization (`1st–99th` pct) or imputation.
- **In-Fold Preprocessing:** Median imputation, standard scaling, and `OneHotEncoder(handle_unknown="ignore")` run inside `sklearn.Pipeline` + `StratifiedKFold(5)`.
- **Regulatory Auditability:** RBI/Basel-aligned explanations via SHAP `TreeExplainer` & `LinearExplainer` for both global policy and individual rejections.
                """
            )

    st.space("small")
    st.subheader("System Architecture")

    with st.grid(4):
        with st.container(border=True):
            st.markdown("**:material/fact_check: 1. Data Audit & Cleaning**")
            st.caption(
                "Scanned 25,000 rows x 22 columns, verified 0 duplicates, nullified impossible bounds, "
                "removed leakage, and applied train-fitted winsorization."
            )
        with st.container(border=True):
            st.markdown("**:material/calculate: 2. Indian Banking Ratios**")
            st.caption(
                "Engineered 7 domain features: FOIR (`total_obligations_ratio`), `emi_to_income_ratio`, "
                "`loan_to_annual_income_ratio`, `credit_hunger_index`, and CIBIL bands."
            )
        with st.container(border=True):
            st.markdown("**:material/model_training: 3. Multi-Model Benchmark**")
            st.caption(
                "Evaluated Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting "
                "on identical 5-fold stratified CV splits and held-out test set."
            )
        with st.container(border=True):
            st.markdown("**:material/psychology: 4. Cost Tuning & SHAP**")
            st.caption(
                "Swept probability thresholds (0.05–0.80) under a 5:1 FN:FP cost matrix and generated "
                "global + per-applicant SHAP waterfall explanations."
            )

    st.space("small")
    with st.container(border=True):
        st.subheader("Key Deliverables & Resume Highlights")
        st.markdown(
            """
- **Notebooks:** Self-contained, fully executed [01_EDA_and_Data_Cleaning.ipynb](notebooks/01_EDA_and_Data_Cleaning.ipynb) and [02_Modelling_Benchmarking_SHAP.ipynb](notebooks/02_Modelling_Benchmarking_SHAP.ipynb) with business commentary before every code block.
- **Best Ranking Model:** `Logistic Regression` (`class_weight='balanced'`) achieved **0.8728 ROC-AUC**, **0.3832 PR-AUC**, and **0.7883 Recall** in **0.22s** fit time.
- **Best F1 / Precision Models:** `Random Forest` achieved the highest **F1-Score (0.3940)**, while `HistGradientBoosting` achieved the highest **Precision (0.6567)** at the default 0.50 cutoff.
- **Top Risk Drivers (SHAP):** `cibil_score`, `late_payments_last_12m`, `total_obligations_ratio` (FOIR), `emi_to_income_ratio`, and `credit_utilization_ratio_pct`.
            """
        )


# ─────────────────────────────────────────────────────────────
# PAGE 2 — EDA & Business Risk Insights
# ─────────────────────────────────────────────────────────────
def page_eda() -> None:
    st.title("Exploratory Data Analysis & Risk Insights")
    st.caption("Six focused analyses translating raw borrower attributes into underwriting insights.")

    view = st.segmented_control(
        "Analysis Focus",
        ["CIBIL & Pricing", "Borrower Segments", "Behavioural Signals", "Correlations"],
        default="CIBIL & Pricing",
    )

    if view == "CIBIL & Pricing":
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            with st.container(border=True):
                st.subheader("Default Rate by CIBIL Score Band")
                show_result_image(
                    "eda_01_cibil_default_rate.png",
                    "CIBIL <650 borrowers show sharply elevated default rates vs 750+ prime borrowers.",
                    "Bar chart of default rate by CIBIL score band",
                )
        with col2:
            with st.container(border=True):
                st.subheader("Interest Rate Pricing Across CIBIL Bands")
                show_result_image(
                    "eda_05_interest_rate_cibil.png",
                    "Defaulters are offered higher median interest rates even within the same CIBIL band.",
                    "Grouped bar chart of median interest rate by CIBIL band and default status",
                )
        st.info(
            "CIBIL score is the single strongest separator of credit risk, and existing lender interest-rate pricing "
            "confirms economically coherent risk signal in the dataset.",
            title="Underwriting takeaway",
        )

    elif view == "Borrower Segments":
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            with st.container(border=True):
                st.subheader("Employment Type & City Tier")
                show_result_image(
                    "eda_02_employment_citytier.png",
                    "Self-employed / variable-income profiles and Tier-3 borrowers exhibit higher default rates.",
                    "Default rate by employment type and city tier",
                )
        with col2:
            with st.container(border=True):
                st.subheader("Loan Size & Income Distributions (Log Scale)")
                show_result_image(
                    "eda_03_loan_income_distribution.png",
                    "Income and loan size overlap heavily — linear/non-linear multi-feature combinations are required.",
                    "Log-scale histograms of loan amount and monthly income by default status",
                )

    elif view == "Behavioural Signals":
        with st.container(border=True):
            st.subheader("Credit Utilisation vs Late Payments")
            show_result_image(
                "eda_04_utilization_late_payments.png",
                "Defaults concentrate in the upper-right region (high credit utilisation + repeated delinquencies).",
                "Scatter plot of credit utilization versus late payments colored by default flag",
            )
        st.info(
            "Combining credit utilization, recent inquiries, and CIBIL deficit motivated the engineered `credit_hunger_index` feature.",
            title="Feature engineering link",
        )

    else:
        with st.container(border=True):
            st.subheader("Numerical Feature Correlation Matrix")
            show_result_image(
                "eda_06_correlation_heatmap.png",
                "Pearson correlations across numerical features and default_flag.",
                "Correlation heatmap of numerical features",
            )


# ─────────────────────────────────────────────────────────────
# PAGE 3 — Multi-Model Benchmarking
# ─────────────────────────────────────────────────────────────
def page_benchmark() -> None:
    st.title("Multi-Model Benchmarking")
    st.caption("4 classifier families evaluated on identical 5-fold stratified CV splits and held-out 5,000-row test set.")

    test_df = load_csv("test_benchmark.csv")
    cv_df = load_csv("cv_benchmark.csv")

    if test_df is not None:
        st.subheader("Held-Out Test Set Performance (n = 5,000)")
        metric_cols = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1-Score"]
        time_cols = ["Fit Time(s)", "Pred Time(s)"]
        styled = (
            test_df.style.format({c: "{:.4f}" for c in metric_cols + ["Pred Time(s)"]})
            .format({"Fit Time(s)": "{:.2f}"})
            .highlight_max(
                subset=metric_cols,
                props="background-color: #196f3d; color: #ffffff; font-weight: 700;",
            )
            .highlight_min(
                subset=time_cols,
                props="background-color: #196f3d; color: #ffffff; font-weight: 700;",
            )
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption("Highlighted green cells mark the best value in each column (highest for scores, lowest for latency).")
    else:
        st.warning("`results/test_benchmark.csv` not found. Run Notebook 2 first.", title="Benchmark missing")

    if cv_df is not None:
        with st.expander("5-Fold Stratified Cross-Validation Stability (Mean +/- Std)", expanded=False):
            cv_display = pd.DataFrame({"Model": cv_df["Model"]})
            for raw, label in [
                ("roc_auc", "ROC-AUC"),
                ("average_precision", "PR-AUC"),
                ("recall", "Recall"),
                ("precision", "Precision"),
                ("f1", "F1-Score"),
            ]:
                cv_display[label] = [
                    f"{m:.4f} ± {s:.4f}"
                    for m, s in zip(cv_df[f"{raw}_mean"], cv_df[f"{raw}_std"])
                ]
            cv_display["CV Time (s)"] = cv_df["cv_time_sec"].map(lambda v: f"{v:.2f}")
            st.dataframe(cv_display, width="stretch", hide_index=True)

    st.space("small")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.subheader("ROC Curves")
            show_result_image(
                "roc_curves.png",
                "Logistic Regression (0.873) and HistGB (0.872) lead ranking performance across thresholds.",
                "ROC curves comparing all four models",
            )
    with col2:
        with st.container(border=True):
            st.subheader("Precision-Recall Curves")
            show_result_image(
                "pr_curves.png",
                "PR-AUC focuses on the 6.14% default class; Logistic Regression (0.383) is 6.2x random baseline.",
                "Precision-Recall curves comparing all four models",
            )

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        with st.container(border=True):
            st.subheader("Metric Comparison")
            show_result_image(
                "benchmark_bar.png",
                "Side-by-side comparison of ROC-AUC, PR-AUC, Recall, Precision, and F1.",
                "Grouped bar chart of model metrics",
            )
    with col4:
        with st.container(border=True):
            st.subheader("Confusion Matrices (Default t = 0.50)")
            show_result_image(
                "confusion_matrices.png",
                "HistGB without class_weight has high precision (0.657) but misses 85.7% of defaults at t=0.50.",
                "Confusion matrices for all four models at 0.50 threshold",
            )


# ─────────────────────────────────────────────────────────────
# PAGE 4 — Threshold Tuning, SHAP & Error Analysis
# ─────────────────────────────────────────────────────────────
def page_explainability() -> None:
    st.title("Threshold Optimization, SHAP & Error Analysis")
    st.caption("Converting probability scores into cost-optimal, regulator-auditable lending decisions.")

    tab_thresh, tab_shap, tab_err = st.tabs([
        ":material/tune: Threshold Optimization",
        ":material/psychology: SHAP Explainability",
        ":material/troubleshoot: Error Analysis",
    ])

    with tab_thresh:
        thresh_df = load_csv("threshold_summary.csv")
        if thresh_df is not None and not thresh_df.empty:
            row = thresh_df.iloc[0]
            base_cost = float(row["baseline_cost_at_0.50"])
            opt_cost = float(row["cost_at_opt_cost"])
            saving_pct = (base_cost - opt_cost) / base_cost * 100 if base_cost else 0.0

            with st.grid(4):
                st.metric("Model Tuned", str(row["model"]), "Cost matrix FN:FP = 5:1", border=True)
                st.metric("Cost-Optimal Cutoff", f"{row['threshold_min_cost']:.3f}", f"Recall {row['recall_at_opt_cost']:.4f}", border=True)
                st.metric("Max-F1 Cutoff", f"{row['threshold_max_f1']:.2f}", f"F1 {row['f1_at_opt_f1']:.4f}", border=True)
                st.metric("Expected Cost Saving", f"-{saving_pct:.1f}%", f"{base_cost:.0f} -> {opt_cost:.0f}", border=True)

        with st.container(border=True):
            show_result_image(
                "threshold_analysis_Logistic_Regression.png",
                "Sweeping decision thresholds from 0.05 to 0.80 aligns predictions with asymmetric lending losses.",
                "Precision, recall, F1, and business cost curves across thresholds",
            )

    with tab_shap:
        shap_options = {
            "Random Forest": "Random_Forest",
            "Gradient Boosting (HistGB)": "Gradient_Boosting_HistGB",
            "Logistic Regression": "Logistic_Regression",
        }
        chosen_shap = st.segmented_control(
            "Select Model for SHAP Inspection",
            list(shap_options.keys()),
            default="Random Forest",
        )
        prefix = shap_options[chosen_shap]

        shap_imp = load_csv(f"shap_importance_{prefix}.csv")
        col1, col2 = st.columns([1.3, 1.0], gap="medium")
        with col1:
            with st.container(border=True):
                st.subheader("Global Impact Direction (Beeswarm)")
                show_result_image(
                    f"shap_beeswarm_{prefix}.png",
                    f"SHAP beeswarm for {chosen_shap} — red = high feature value, positive = pushes toward default.",
                    f"SHAP beeswarm plot for {chosen_shap}",
                )
        with col2:
            with st.container(border=True):
                st.subheader("Top 15 Risk Drivers")
                if shap_imp is not None:
                    top15 = shap_imp.head(15).copy()
                    top15["Mean_SHAP"] = top15["Mean_SHAP"].map(lambda x: round(float(x), 4))
                    st.dataframe(top15, width="stretch", hide_index=True, height=460)
                else:
                    show_result_image(
                        f"shap_bar_{prefix}.png",
                        f"Mean |SHAP| feature importance for {chosen_shap}.",
                        f"SHAP bar plot for {chosen_shap}",
                    )

        st.space("small")
        st.subheader("Individual Applicant Waterfall Explanations")
        w1, w2 = st.columns(2, gap="medium")
        with w1:
            with st.container(border=True):
                st.markdown("**High-Risk Applicant (True Positive)**")
                show_result_image(
                    f"shap_waterfall_{prefix}_HighRisk_TP.png",
                    "Features pushing predicted default risk above the base rate.",
                    f"High-risk applicant SHAP waterfall for {chosen_shap}",
                )
        with w2:
            with st.container(border=True):
                st.markdown("**Low-Risk Applicant (True Negative)**")
                show_result_image(
                    f"shap_waterfall_{prefix}_LowRisk_TN.png",
                    "Strong CIBIL and clean repayment history pushing risk down.",
                    f"Low-risk applicant SHAP waterfall for {chosen_shap}",
                )

    with tab_err:
        st.subheader("False Negatives vs True Positives (Logistic Regression)")
        fn_df = load_csv("error_FN_profile_Logistic_Regression.csv")
        tp_df = load_csv("error_TP_profile_Logistic_Regression.csv")

        if fn_df is not None and tp_df is not None:
            fn_idx = fn_df.set_index(fn_df.columns[0])
            tp_idx = tp_df.set_index(tp_df.columns[0])
            key_feats = [
                "cibil_score",
                "late_payments_last_12m",
                "credit_utilization_ratio_pct",
                "monthly_income_inr",
                "existing_emi_monthly_inr",
                "loan_amount_requested_inr",
            ]
            rows = []
            for feat in key_feats:
                if feat in fn_idx.index and feat in tp_idx.index:
                    fn_m = float(fn_idx.loc[feat, "mean"])
                    tp_m = float(tp_idx.loc[feat, "mean"])
                    rows.append({
                        "Feature": feat,
                        "FN Mean (Missed Defaults)": round(fn_m, 1),
                        "TP Mean (Caught Defaults)": round(tp_m, 1),
                        "Diff (FN - TP)": round(fn_m - tp_m, 1),
                    })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.info(
            "Missed defaulters (False Negatives) tend to have slightly higher CIBIL scores and fewer recent "
            "late payments — borderline borrowers who appear creditworthy on static bureau snapshots before "
            "experiencing an income shock. Incorporating bank-statement cash-flow trends is the natural next step.",
            title="Key error-analysis finding",
        )


# ─────────────────────────────────────────────────────────────
# PAGE 5 — Live Underwriting Scorer
# ─────────────────────────────────────────────────────────────
PRESETS = {
    "Prime Salaried": {
        "age": 33, "gender": "Female", "marital_status": "Married", "city_tier": "Tier 1",
        "state": "Maharashtra", "employment_type": "Salaried - PSU", "monthly_income_inr": 85000,
        "bank_account_vintage": 7.5, "existing_loans_count": 1, "existing_emi": 8000,
        "credit_utilization": 22.0, "num_inquiries": 0, "late_payments": 0,
        "cibil_score": 785, "loan_purpose": "Home Renovation", "loan_amount": 250000,
        "loan_tenure": 36, "interest_rate": 12.5, "collateral": "Yes",
    },
    "Borderline FOIR": {
        "age": 29, "gender": "Male", "marital_status": "Single", "city_tier": "Tier 2",
        "state": "Karnataka", "employment_type": "Salaried - Private", "monthly_income_inr": 40000,
        "bank_account_vintage": 2.5, "existing_loans_count": 2, "existing_emi": 15000,
        "credit_utilization": 52.0, "num_inquiries": 2, "late_payments": 1,
        "cibil_score": 665, "loan_purpose": "Debt Consolidation", "loan_amount": 220000,
        "loan_tenure": 24, "interest_rate": 19.5, "collateral": "No",
    },
    "High-Risk Delinquent": {
        "age": 26, "gender": "Male", "marital_status": "Single", "city_tier": "Tier 3",
        "state": "Uttar Pradesh", "employment_type": "Freelancer", "monthly_income_inr": 22000,
        "bank_account_vintage": 1.0, "existing_loans_count": 4, "existing_emi": 11000,
        "credit_utilization": 84.0, "num_inquiries": 5, "late_payments": 4,
        "cibil_score": 540, "loan_purpose": "Personal Expense", "loan_amount": 180000,
        "loan_tenure": 24, "interest_rate": 24.5, "collateral": "No",
    },
}


def page_scorer() -> None:
    st.title("Live Underwriting Risk Scorer")
    st.caption("Score any borrower profile in real time across all 4 trained pipelines with dynamic threshold control.")

    models = load_models()
    if not models:
        st.error(
            "No trained `.joblib` pipelines found in `models/`. Run `notebooks/02_Modelling_Benchmarking_SHAP.ipynb` first.",
            title="Models unavailable",
        )
        return

    thresh_df = load_csv("threshold_summary.csv")
    default_opt = 0.694
    if thresh_df is not None and "threshold_min_cost" in thresh_df.columns:
        default_opt = float(thresh_df["threshold_min_cost"].iloc[0])

    # Top controls
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.3], gap="medium")
    with ctrl1:
        default_idx = list(models.keys()).index("Logistic Regression") if "Logistic Regression" in models else 0
        model_name = st.selectbox("Scoring Pipeline", list(models.keys()), index=default_idx)
    with ctrl2:
        preset_name = st.segmented_control(
            "Quick Test Profile",
            list(PRESETS.keys()),
            default="Borderline FOIR",
        )
    with ctrl3:
        threshold = st.slider(
            "Decision Threshold (Reject >= Cutoff)",
            min_value=0.05,
            max_value=0.85,
            value=default_opt,
            step=0.01,
            help="Default 0.694 minimizes expected 5:1 FN:FP business cost for Logistic Regression.",
        )

    p = PRESETS[preset_name or "Borderline FOIR"]
    pipeline = models[model_name]

    with st.form("applicant_form"):
        st.subheader("Borrower & Credit Bureau Profile")
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            age = st.number_input("Age", 18, 75, p["age"])
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(p["gender"]))
            marital_opts = ["Single", "Married", "Divorced", "Widowed"]
            marital_status = st.selectbox("Marital Status", marital_opts, index=marital_opts.index(p["marital_status"]))
            tier_opts = ["Tier 1", "Tier 2", "Tier 3"]
            city_tier = st.selectbox("City Tier", tier_opts, index=tier_opts.index(p["city_tier"]))
            state_opts = [
                "Maharashtra", "Delhi NCR", "Karnataka", "Tamil Nadu", "Gujarat",
                "Rajasthan", "Uttar Pradesh", "Haryana", "West Bengal", "Telangana",
                "Kerala", "Madhya Pradesh", "Punjab", "Bihar", "Andhra Pradesh",
            ]
            state = st.selectbox("State", state_opts, index=state_opts.index(p["state"]))

        with c2:
            emp_opts = [
                "Salaried - Private", "Salaried - PSU", "Salaried - Government",
                "Business Owner", "Self-Employed Professional", "Freelancer",
            ]
            employment_type = st.selectbox("Employment Type", emp_opts, index=emp_opts.index(p["employment_type"]))
            monthly_income_inr = st.number_input("Monthly Income (INR)", 5000, 1000000, p["monthly_income_inr"], 1000)
            bank_account_vintage = st.slider("Bank Account Vintage (years)", 0.0, 25.0, float(p["bank_account_vintage"]), 0.5)
            existing_loans_count = st.number_input("Existing Loans Count", 0, 10, p["existing_loans_count"])
            existing_emi = st.number_input("Existing EMI / Month (INR)", 0, 250000, p["existing_emi"], 500)

        with c3:
            credit_utilization = st.slider("Credit Utilisation (%)", 0.0, 100.0, float(p["credit_utilization"]), 0.5)
            num_inquiries = st.number_input("Credit Inquiries (last 6m)", 0, 20, p["num_inquiries"])
            late_payments = st.number_input("Late Payments (last 12m)", 0, 20, p["late_payments"])
            cibil_score = st.number_input("CIBIL Score", 300, 900, p["cibil_score"])
            purpose_opts = [
                "Home Renovation", "Medical Emergency", "Education", "Wedding",
                "Vehicle Purchase", "Debt Consolidation", "Business Expansion", "Personal Expense",
            ]
            loan_purpose = st.selectbox("Loan Purpose", purpose_opts, index=purpose_opts.index(p["loan_purpose"]))

        st.subheader("Requested Loan Terms")
        l1, l2, l3, l4 = st.columns(4, gap="medium")
        with l1:
            loan_amount = st.number_input("Loan Amount (INR)", 10000, 5000000, p["loan_amount"], 10000)
        with l2:
            tenure_opts = [12, 24, 36, 48, 60, 72, 84]
            loan_tenure = st.selectbox("Tenure (months)", tenure_opts, index=tenure_opts.index(p["loan_tenure"]))
        with l3:
            interest_rate = st.number_input("Interest Rate (%)", 7.0, 36.0, float(p["interest_rate"]), 0.1)
        with l4:
            collateral = st.selectbox("Collateral Provided", ["No", "Yes"], index=["No", "Yes"].index(p["collateral"]))

        submitted = st.form_submit_button("Run Credit Risk Assessment", type="primary", width="stretch")

    if submitted:
        applicant = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "city_tier": city_tier,
            "state": state,
            "employment_type": employment_type,
            "bank_account_vintage_years": bank_account_vintage,
            "monthly_income_inr": monthly_income_inr,
            "existing_loans_count": existing_loans_count,
            "existing_emi_monthly_inr": existing_emi,
            "credit_utilization_ratio_pct": credit_utilization,
            "num_credit_inquiries_last_6m": num_inquiries,
            "late_payments_last_12m": late_payments,
            "loan_amount_requested_inr": loan_amount,
            "loan_tenure_months": loan_tenure,
            "loan_purpose": loan_purpose,
            "collateral_provided": collateral,
            "cibil_score": cibil_score,
            "interest_rate_offered_pct": interest_rate,
        }])

        prob = float(pipeline.predict_proba(applicant)[0, 1])
        rejected = prob >= threshold
        tier_label, _ = classify_risk_tier(prob)

        engineered = add_financial_ratios(applicant).iloc[0]
        foir = float(engineered["total_obligations_ratio"])
        lti = float(engineered["loan_to_annual_income_ratio"])
        new_emi_ratio = float(engineered["installment_to_income_ratio"])
        hunger = float(engineered["credit_hunger_index"])

        st.space("small")
        with st.grid(4):
            st.metric("Default Probability", f"{prob:.1%}", f"Risk Tier: {tier_label}", border=True)
            st.metric("Underwriting Decision", "REJECT" if rejected else "APPROVE", f"Cutoff: {threshold:.2f}", border=True)
            st.metric("Full FOIR", f"{foir:.1%}", "Above 50% limit" if foir > 0.50 else "Within 50% limit", delta_color="inverse", border=True)
            st.metric("Loan / Annual Income", f"{lti:.2f}x", f"Credit Hunger: {hunger:.2f}", border=True)

        if rejected:
            st.error(
                f"Predicted default probability **{prob:.1%}** meets or exceeds the decision cutoff **{threshold:.2f}**. "
                "Recommend **manual underwriting review or rejection**.",
                title="High Default Risk Flagged",
            )
        else:
            st.success(
                f"Predicted default probability **{prob:.1%}** is below the decision cutoff **{threshold:.2f}**. "
                "Applicant meets automated approval criteria.",
                title="Creditworthy Profile",
            )

        with st.container(border=True):
            st.subheader("Underwriting Signal Breakdown")
            signals = pd.DataFrame([
                {"Signal": "CIBIL Score", "Applicant Value": f"{cibil_score}", "Benchmark": ">= 700 Prime", "Status": "Favorable" if cibil_score >= 700 else "Elevated Risk"},
                {"Signal": "Late Payments (12m)", "Applicant Value": f"{late_payments}", "Benchmark": "0 Delinquencies", "Status": "Clean" if late_payments == 0 else "Delinquency Flag"},
                {"Signal": "Full FOIR (Existing + New EMI)", "Applicant Value": f"{foir:.1%}", "Benchmark": "<= 50% Bank Limit", "Status": "Within Limit" if foir <= 0.50 else "Over-Leveraged"},
                {"Signal": "New EMI / Monthly Income", "Applicant Value": f"{new_emi_ratio:.1%}", "Benchmark": "<= 35%", "Status": "Manageable" if new_emi_ratio <= 0.35 else "High Burden"},
                {"Signal": "Credit Utilisation", "Applicant Value": f"{credit_utilization:.1f}%", "Benchmark": "<= 40%", "Status": "Healthy" if credit_utilization <= 40 else "High Utilisation"},
                {"Signal": "Credit Hunger Index", "Applicant Value": f"{hunger:.2f}", "Benchmark": "< 3.0", "Status": "Normal" if hunger < 3.0 else "Credit Hungry"},
            ])
            st.dataframe(signals, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────
# Navigation & Sidebar Metadata
# ─────────────────────────────────────────────────────────────
pages = [
    st.Page(page_overview, title="Executive Summary", icon=":material/dashboard:", default=True),
    st.Page(page_eda, title="EDA & Risk Insights", icon=":material/analytics:"),
    st.Page(page_benchmark, title="Model Benchmark", icon=":material/compare:"),
    st.Page(page_explainability, title="Threshold & SHAP", icon=":material/tune:"),
    st.Page(page_scorer, title="Live Scorer", icon=":material/bolt:"),
]

with st.sidebar:
    st.markdown("### Credit Risk Engine")
    st.caption("India Personal Loan Default · 2026")
    st.divider()
    st.markdown(
        """
**Portfolio Highlights**
- `25,000` borrower profiles
- `6.14%` default rate (`1:15.3`)
- `7` Indian banking ratios
- `4` benchmarked classifiers
- `SHAP` + `5:1` cost tuning
        """
    )
    st.caption("Built by Kartik Kumar Singh · Python 3.14 · Scikit-learn · SHAP")

nav = st.navigation(pages)
nav.run()
