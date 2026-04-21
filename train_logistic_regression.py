import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.db.clickhouse_client import get_client

print("=== Training Logistic Regression ===")

RANDOM_SEED = 42

client = get_client()

feature_names = [
    "group_popularity",
    "log_group_popularity",
    "user_based_score",
    "item_based_score",
    "max_group_similarity",
    "sum_group_similarity",
    "common_members_with_profile",
    "is_in_both_recs",
]

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

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_SEED,
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

X_train = train_df[feature_names]
y_train = train_df["label"]
X_test = test_df[feature_names]
y_test = test_df["label"]

train_groups = train_df["user_id"]
test_groups = test_df["user_id"]

print(f"Train rows: {len(train_df)}, users: {train_groups.nunique()}")
print(f"Test rows: {len(test_df)}, users: {test_groups.nunique()}")

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

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

test_auc = roc_auc_score(y_test, y_pred_proba)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)

print("\n--- Final classification metrics on held-out users ---")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0))

joblib.dump(model, "recommendation_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("\nModel, scaler and feature_names saved successfully.")

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