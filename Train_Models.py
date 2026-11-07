"""
Bank Marketing (UCI / Moro et al., 2011): train 5 classifiers on the same data,
evaluate them, and save models + test split for the Streamlit app.

Run:  python train_models.py
"""
import os, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIG ----------------
DATA_PATH   = "1/bank.csv"   # 4,521 rows, semicolon-separated
SEP         = ";"
TARGET_COL  = "y"
DROP_COLS   = ["duration"]      # leakage: only known after the call ends
TEST_SIZE   = 0.2
SEED        = 42
# ----------------------------------------

def compute_metrics(y_true, y_pred, y_proba, n_classes):
    avg = "binary" if n_classes == 2 else "weighted"
    auc = (roc_auc_score(y_true, y_proba[:, 1]) if n_classes == 2
           else roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }

def main():
    os.makedirs("model", exist_ok=True)
    df = pd.read_csv(DATA_PATH, sep=SEP).drop_duplicates()
    df = df.drop(columns=DROP_COLS)
    print("Shape after cleaning:", df.shape)
    print("Target balance:\n", df[TARGET_COL].value_counts())

    X, y_raw = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
    le = LabelEncoder().fit(y_raw)            # 'no' -> 0, 'yes' -> 1
    y = le.transform(y_raw)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    models = {
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree":          DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "KNN":                    KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes":            GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=SEED),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)
        m = compute_metrics(y_test, pred, proba, n_classes)
        rows.append({"ML Model Name": name, **m})
        joblib.dump(pipe, f"model/{name.replace(' ', '_').lower()}.joblib")
        print(f"{name:24s}", {k: round(v, 4) for k, v in m.items()})

    res = pd.DataFrame(rows).round(4)
    res.to_csv("model/metrics_comparison.csv", index=False)
    joblib.dump(le, "model/label_encoder.joblib")
    with open("model/config.txt", "w") as f:
        f.write(TARGET_COL)

    # Held-out test split (original labels, comma-separated) for the app upload
    test_df = X_test.copy()
    test_df[TARGET_COL] = le.inverse_transform(y_test)
    test_df.to_csv("test_data.csv", index=False)
    print("\nTest rows saved:", len(test_df))
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()