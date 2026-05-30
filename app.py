import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Load Data & Models ---
df_raw = pd.read_csv("data.csv")
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Prepare X, y for model comparison ---
X = df_raw.drop(columns=["customer_id", "delinquent_account", "index"], errors="ignore")
y = df_raw["delinquent_account"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- App Config ---
st.set_page_config(page_title="Delinquency Dashboard", layout="wide")
page = st.sidebar.selectbox("Navigate", ["EDA", "Model Comparison", "Prediction"])

# ============================================================
# PAGE 1 - EDA
# ============================================================
if page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("Understanding key risk factors through visualization.")

    # Payment History
    st.subheader("Payment History (Month 1-6)")
    month_cols = ["month_1", "month_2", "month_3", "month_4", "month_5", "month_6"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, col in enumerate(month_cols):
        ax = axes[i // 3, i % 3]
        df_raw[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Payment History")
    fig.tight_layout()
    st.pyplot(fig)
    st.info("The late payment category shows the most variation across six months. All three categories cluster within the 150–175 band. Month 5 shows an unusual spike in missed payments worth investigating.")

    # Employment Status
    st.subheader("Employment Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    employment_cols = [c for c in df_raw.columns if "employment_status" in c]
    emp_counts = df_raw[employment_cols].sum().rename(lambda x: x.replace("employment_status_", ""))
    emp_counts.plot(kind="bar", ax=ax)
    ax.set_title("Employment Status")
    ax.set_xlabel("")
    st.pyplot(fig)
    st.info("Employed customers (~240) dominate the dataset. Self-employed, unemployed and retired customers cluster between 80–95.")

    # DTI
    st.subheader("Debt to Income Ratio")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_raw, x="debt_to_income_ratio", bins=20, kde=True, ax=ax)
    ax.set_title("Debt to Income Ratio")
    st.pyplot(fig)
    st.info("Peak around 0.30–0.32 — moderate risk zone. Small high-risk cluster above 0.4.")

    # Income
    st.subheader("Income Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_raw, x="income", bins=20, kde=True, ax=ax)
    ax.set_title("Income")
    st.pyplot(fig)
    st.info("Unusual spike at ₹100,000 — possible data clustering at a round number. Right skewed with a consistent cluster between 150k–200k.")

    # Age
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_raw, x="age", bins=20, kde=True, ax=ax)
    ax.set_title("Age")
    st.pyplot(fig)
    st.info("Right skewed. Bulk of borrowers are younger (20–50). Older customers (60–70) represent a smaller but potentially more reliable segment.")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = ["age", "income", "debt_to_income_ratio", "credit_score", "credit_utilization"]
    corr = df_raw[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.info("All numeric features show near-zero correlation with each other — no multicollinearity, but also weak linear relationships suggesting non-linear patterns.")

# ============================================================
# PAGE 2 - MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Comparing Logistic Regression, Random Forest and XGBoost on delinquency prediction.")

    # LR predictions
    smote = SMOTE(random_state=42)
    X_train_sampled, y_train_sampled = smote.fit_resample(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    # RF predictions
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.3).astype(int)

    # XGB predictions
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.3).astype(int)

    models = {
        "Logistic Regression": (lr_pred, lr_proba),
        "Random Forest": (rf_pred, rf_proba),
        "XGBoost": (xgb_pred, xgb_proba)
    }

    # Metrics table
    st.subheader("Model Metrics Summary")
    rows = []
    for name, (pred, proba) in models.items():
        report = classification_report(y_test, pred, output_dict=True)
        rows.append({
            "Model": name,
            "Accuracy": round(report["accuracy"], 2),
            "Precision (Delinquent)": round(report["1"]["precision"], 2),
            "Recall (Delinquent)": round(report["1"]["recall"], 2),
            "F1 (Delinquent)": round(report["1"]["f1-score"], 2)
        })
    st.dataframe(pd.DataFrame(rows))

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (pred, _)) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (_, proba) in models.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    st.pyplot(fig)
    st.warning("All three models show AUC near 0.5 — essentially random. This confirms the dataset lacks sufficient predictive signal for reliable delinquency classification.")

# ============================================================
# PAGE 3 - PREDICTION
# ============================================================
elif page == "Prediction":
    st.title("Delinquency Prediction")
    st.markdown("Enter customer details to predict delinquency probability.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Income (₹)", 10000, 500000, 100000)
        credit_score = st.slider("Credit Score", 300, 850, 600)
        credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.4)
        missed_payments = st.slider("Missed Payments", 0, 10, 2)
        loan_balance = st.number_input("Loan Balance (₹)", 0, 200000, 50000)
        dti = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)
        account_tenure = st.slider("Account Tenure (years)", 0, 20, 5)
        credit_card_type = st.selectbox("Credit Card Type", ["student", "standard", "platinum", "gold", "business"])
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    with col2:
        month_vals = {}
        for m in ["month_1", "month_2", "month_3", "month_4", "month_5", "month_6"]:
            month_vals[m] = st.selectbox(m.replace("_", " ").title(), [2, 1, 0], format_func=lambda x: {2: "On-time", 1: "Late", 0: "Missed"}[x])

        employment = st.selectbox("Employment Status", ["employed", "retired", "self-employed", "unemployed"])
        location = st.selectbox("Location", ["chicago", "houston", "los angeles", "new york", "phoenix"])

    card_map = {"student": 0, "standard": 1, "platinum": 2, "gold": 3, "business": 4}

    input_data = {
        "age": age, "income": income, "credit_score": credit_score,
        "credit_utilization": credit_utilization, "missed_payments": missed_payments,
        "loan_balance": loan_balance, "debt_to_income_ratio": dti,
        "account_tenure": account_tenure, "credit_card_type": card_map[credit_card_type],
        **month_vals,
        "employment_status_employed": 1 if employment == "employed" else 0,
        "employment_status_retired": 1 if employment == "retired" else 0,
        "employment_status_self-employed": 1 if employment == "self-employed" else 0,
        "employment_status_unemployed": 1 if employment == "unemployed" else 0,
        "location_chicago": 1 if location == "chicago" else 0,
        "location_houston": 1 if location == "houston" else 0,
        "location_los angeles": 1 if location == "los angeles" else 0,
        "location_new york": 1 if location == "new york" else 0,
        "location_phoenix": 1 if location == "phoenix" else 0,
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        if model_choice == "Logistic Regression":
            input_scaled = scaler.transform(input_df)
            proba = lr_model.predict_proba(input_scaled)[0][1]
        elif model_choice == "Random Forest":
            proba = rf_model.predict_proba(input_df)[0][1]
        else:
            proba = xgb_model.predict_proba(input_df)[0][1]

        st.metric("Delinquency Probability", f"{proba:.1%}")

        if proba >= 0.5:
            st.error("High Risk — Likely Delinquent")
        elif proba >= 0.3:
            st.warning("Moderate Risk — Monitor Closely")
        else:
            st.success("Low Risk — Likely Safe")
