import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

RNG = np.random.default_rng(42)


def generate_dataset(n: int = 2000) -> pd.DataFrame:
    tenure = RNG.integers(1, 73, size=n).astype(int)
    monthly_charges = RNG.uniform(20, 120, size=n).round(2)
    total_charges = (tenure * monthly_charges + RNG.normal(0, 50, n)).clip(0).round(2)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_probs = [0.55, 0.25, 0.20]
    contract = RNG.choice(contract_choices, size=n, p=contract_probs)

    internet_choices = ["Fiber optic", "DSL", "No"]
    internet_probs = [0.45, 0.40, 0.15]
    internet_service = RNG.choice(internet_choices, size=n, p=internet_probs)

    tech_map = {
        "Fiber optic": ["Yes", "No"],
        "DSL": ["Yes", "No"],
        "No": ["No internet service"],
    }
    tech_support = np.array([
        RNG.choice(tech_map[svc]) for svc in internet_service
    ])

    senior_citizen = (RNG.random(n) < 0.16).astype(int)
    dependents = (RNG.random(n) < 0.30).astype(int)
    num_services = RNG.integers(1, 8, size=n).astype(int)
    paperless_billing = (RNG.random(n) < 0.59).astype(int)

    # Churn probability is shaped by real-world drivers
    churn_logit = (
        -2.5
        + 1.8 * (contract == "Month-to-month").astype(float)
        - 0.6 * (contract == "One year").astype(float)
        - 1.5 * (contract == "Two year").astype(float)
        + 0.9 * (internet_service == "Fiber optic").astype(float)
        - 0.4 * (internet_service == "No").astype(float)
        + 0.5 * (monthly_charges > 80).astype(float)
        - 0.04 * tenure
        + 0.4 * (tech_support == "No").astype(float)
        + 0.3 * senior_citizen
        - 0.3 * dependents
        + RNG.normal(0, 0.4, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (RNG.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "senior_citizen": senior_citizen,
        "dependents": dependents,
        "num_services": num_services,
        "paperless_billing": paperless_billing,
        "churn": churn,
    })


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["contract", "internet_service", "tech_support"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    churn_counts = df["churn"].value_counts().sort_index()
    axes[0].bar(["Retained", "Churned"], churn_counts.values,
                color=["#1565C0", "#C62828"], edgecolor="white", width=0.5)
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")
    axes[0].set_title("Churn Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    for churn_val, color, label in [(0, "#1565C0", "Retained"), (1, "#C62828", "Churned")]:
        axes[1].hist(df[df["churn"] == churn_val]["tenure"], bins=30,
                     alpha=0.6, color=color, label=label, edgecolor="white")
    axes[1].set_title("Tenure by Churn Status", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    groups = [df[df["churn"] == 0]["monthly_charges"], df[df["churn"] == 1]["monthly_charges"]]
    bp = axes[2].boxplot(groups, labels=["Retained", "Churned"], patch_artist=True,
                         medianprops={"color": "white", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#1565C0")
    bp["boxes"][1].set_facecolor("#C62828")
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    axes[2].set_title("Monthly Charges by Churn Status", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Monthly Charges ($)")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "eda.png"), dpi=150)
    plt.close()
    print("Saved: eda.png")


def plot_feature_importance(model_pipe, feature_cols: list) -> None:
    rf = model_pipe.named_steps["clf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(feature_cols)))
    ax.barh(range(len(feature_cols)), importances[idx][::-1],
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in idx][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: feature_importance.png")


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def evaluate(name: str, pipe, X_tr, X_te, y_tr, y_te) -> dict:
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="roc_auc")

    print(f"\n{'=' * 55}")
    print(f"{name}")
    print(f"{'=' * 55}")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"5-CV AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(classification_report(y_te, y_pred, target_names=["Retained", "Churned"]))

    return {"name": name, "pipe": pipe, "acc": acc, "auc": auc}


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df_raw = generate_dataset(n=2000)
    print(f"Generated dataset: {df_raw.shape[0]} rows | churn rate: {df_raw['churn'].mean():.2%}")

    plot_eda(df_raw)

    df = encode_categoricals(df_raw)
    feature_cols = [c for c in df.columns if c != "churn"]
    X = df[feature_cols].values
    y = df["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_defs = [
        ("Random Forest", make_pipeline(
            RandomForestClassifier(n_estimators=200, max_depth=8,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
        )),
        ("Gradient Boosting", make_pipeline(
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        )),
        ("Logistic Regression", make_pipeline(
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)
        )),
    ]

    results = []
    for name, pipe in model_defs:
        res = evaluate(name, pipe, X_train, X_test, y_train, y_test)
        results.append(res)

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest model: {best['name']} (AUC={best['auc']:.4f})")

    # Use the RF pipeline for feature importance since it's always trained
    rf_result = next(r for r in results if r["name"] == "Random Forest")
    plot_feature_importance(rf_result["pipe"], feature_cols)

    joblib.dump(best["pipe"], os.path.join(MODEL_DIR, "churn_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "features.pkl"))
    print(f"Saved model artifacts to {MODEL_DIR}/")
