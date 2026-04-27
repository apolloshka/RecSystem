import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.db.clickhouse_client import get_client

from src.recommenders.config import (
    RANDOM_SEED,
    MODEL_FEATURE_NAMES,
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    MODEL_BUNDLE_PATH,
)

print("=== Training Logistic Regression ===")

client = get_client()

feature_names = MODEL_FEATURE_NAMES

columns = ["user_id", "candidate_group_id", "label"] + feature_names + ["created_at"]

query = f"""
    SELECT
        user_id,
        candidate_group_id,
        label,
        {", ".join(feature_names)},
        created_at
    FROM ml_dataset
"""

result = client.query(query)
df = pd.DataFrame(result.result_rows, columns=columns)

if df.empty:
    raise ValueError("ml_dataset is empty. Run build_ml_dataset.py first.")

X = df[feature_names]
y = df["label"]
groups = df["user_id"]

print(f"Dataset: {len(df)} rows, {X.shape[1]} features")
print(f"Unique users: {groups.nunique()}")
print(f"Positive: {(y == 1).sum()}, Negative: {(y == 0).sum()}")

if y.nunique() < 2:
    raise ValueError("Need both classes in ml_dataset to train the model.")

X_train = X
y_train = y
train_groups = groups
print(f"Train rows: {len(X_train)}, users: {train_groups.nunique()}")

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=0.01,
                max_iter=2000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            ),
        ),
    ]
)

unique_train_users = train_groups.nunique()
if unique_train_users >= 2:
    n_splits = min(5, unique_train_users)
    if n_splits >= 2:
        cv = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv.split(X_train, y_train, groups=train_groups),
            scoring="roc_auc",
        )
        print("\n--- Cross-validation (GroupKFold by user_id) ---")
        print(f"ROC-AUC on folds: {cv_scores}")
        print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
else:
    print("\n--- Cross-validation skipped: too few users ---")

pipeline.fit(X_train, y_train)

scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]
y_pred_proba = pipeline.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_pred_proba)
print("\n--- In-sample metric (for sanity only) ---")
print(f"Train ROC-AUC: {train_auc:.4f}")
print("Use evaluate_ranking.py for proper ranking quality on held-out users.")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(feature_names, FEATURE_NAMES_PATH)
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_names": feature_names,
    },
    MODEL_BUNDLE_PATH,
)

print("\nModel, scaler, feature_names and model bundle saved successfully.")

coefficients = model.coef_[0]
importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_importance": np.abs(coefficients),
    }
).sort_values("abs_importance", ascending=False)

print("\nFeature importance (by coefficient magnitude):")
for _, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['coefficient']:+.6f}")