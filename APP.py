import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="Bank Marketing Classifier Explorer", layout="wide")
st.title("Bank Term-Deposit Subscription: Classifier Explorer")
st.caption("Predicts whether a client subscribes to a term deposit. Models are pre-trained; "
           "uploaded data is only used for inference and scoring.")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}
TARGET = open("model/config.txt").read().strip()
le = joblib.load("model/label_encoder.joblib")

@st.cache_resource
def load_model(name):
    return joblib.load(f"model/{MODEL_FILES[name]}.joblib")

# ---- Sidebar controls ----
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES))
uploaded = st.sidebar.file_uploader("Upload test CSV (features + 'y')", type="csv")
use_bundled = False
if uploaded is None and os.path.exists("test_data.csv"):
    use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)

tab1, tab2 = st.tabs(["Evaluate selected model", "Compare all models"])

with tab1:
    source = uploaded if uploaded is not None else ("test_data.csv" if use_bundled else None)
    if source is None:
        st.info("Upload a test CSV or tick 'Use bundled test_data.csv' to begin.")
    else:
        # sep=None lets pandas sniff ';' (original UCI format) or ',' (our export)
        df = pd.read_csv(source, sep=None, engine="python")
        if TARGET not in df.columns:
            st.error(f"Target column '{TARGET}' not found. Columns found: {list(df.columns)}")
            st.stop()

        st.subheader("Data preview")
        st.dataframe(df.head(), use_container_width=True)

        X, y_raw = df.drop(columns=[TARGET]), df[TARGET]
        try:
            y = le.transform(y_raw)
        except ValueError:
            st.error(f"Unknown target labels. Expected one of {list(le.classes_)}.")
            st.stop()

        model = load_model(model_name)
        try:
            pred = model.predict(X)
            proba = model.predict_proba(X)
        except Exception as e:
            st.error(f"Feature mismatch with the trained model: {e}")
            st.stop()

        n = len(le.classes_)
        avg = "binary" if n == 2 else "weighted"
        auc = (roc_auc_score(y, proba[:, 1]) if n == 2
               else roc_auc_score(y, proba, multi_class="ovr", average="weighted"))
        met = {
            "Accuracy": accuracy_score(y, pred),
            "AUC": auc,
            "Precision": precision_score(y, pred, average=avg, zero_division=0),
            "Recall": recall_score(y, pred, average=avg, zero_division=0),
            "F1": f1_score(y, pred, average=avg, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        }

        st.subheader(f"Metrics: {model_name}")
        for col, (k, v) in zip(st.columns(6), met.items()):
            col.metric(k, f"{v:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
        with c2:
            st.subheader("Classification report")
            rep = classification_report(y, pred, target_names=[str(c) for c in le.classes_],
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

with tab2:
    st.subheader("All models on the held-out test split (computed during training)")
    st.dataframe(pd.read_csv("model/metrics_comparison.csv"), use_container_width=True)