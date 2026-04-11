import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.db.clickhouse_client import get_client

print("=== Training Logistic Regression ===")

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

print(f"Dataset: {len(df)} rows, {X.shape[1]} features")
print(f"Positive: {(y == 1).sum()}, Negative: {(y == 0).sum()}")

if y.nunique() < 2:
    raise ValueError("Need both classes in ml_dataset to train the model.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    C=0.1,
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

min_class_count = y_train.value_counts().min()
if min_class_count >= 2:
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=cv_strategy,
            scoring="roc_auc"
        )
        print("\n--- Cross-validation ---")
        print(f"ROC-AUC on folds: {cv_scores}")
        print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
else:
    print("\n--- Cross-validation skipped: too few examples per class ---")

model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

test_auc = roc_auc_score(y_test, y_pred_proba)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)

print("\n--- Final test metrics ---")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0))

joblib.dump(model, "recommendation_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("\n Model, scaler and feature_names saved successfully.")

coefficients = model.coef_[0]
importance_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_importance": np.abs(coefficients)
}).sort_values("abs_importance", ascending=False)

print("\nFeature importance (by coefficient magnitude):")
for _, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['coefficient']:+.6f}")