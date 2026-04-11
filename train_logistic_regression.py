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
   # "user_based_score",
   # "item_based_score",
    "max_group_similarity",
    "sum_group_similarity",
    "common_members_with_profile",
   # "is_in_both_recs",
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

X = df[feature_names] # колонки признаков
y = df["label"] # колонки меток

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
    C=0.01, # регуляризация
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

# кросс-валидация
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

# обучение финальной модели
# p = 1 / (1 + e^-(w₁x₁ + w₂x₂ + ... + w₈x₈ + b)) - вероятность 
# w_new = w_old + learning_rate * error * x * p * (1-p) - изменение веса если модель ошиблась
model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # вероятности
y_pred = model.predict(X_test_scaled) # классы

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

def hit_rate_at_k_per_user(user_preds, k=5):
    sorted_preds = sorted(user_preds, key=lambda x: x[1], reverse=True)
    top_k = sorted_preds[:k]
    return 1 if any(label == 1 for label, _ in top_k) else 0

def ndcg_at_k_per_user(user_preds, k=5):

    sorted_preds = sorted(user_preds, key=lambda x: x[1], reverse=True)
    top_k = sorted_preds[:k]
    
    dcg = 0.0
    for i, (label, _) in enumerate(top_k):
        if label == 1:
            dcg += 1.0 / np.log2(i + 2)
    
    total_positives = sum(1 for label, _ in user_preds if label == 1)
    idcg = 0.0
    for i in range(min(k, total_positives)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

test_indices = X_test.index

user_test = df.loc[test_indices, "user_id"].values

test_df = pd.DataFrame({
    'user_id': user_test,
    'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
    'y_pred_proba': y_pred_proba
})

print("\n--- Ranking metrics (HitRate & NDCG per user, averaged) ---")
for k in [1, 3, 5, 10]:
    hit_rates = []
    ndcgs = []
    
    for user_id, group in test_df.groupby('user_id'):
        user_preds = list(zip(group['y_true'], group['y_pred_proba']))
        hit_rates.append(hit_rate_at_k_per_user(user_preds, k=k))
        ndcgs.append(ndcg_at_k_per_user(user_preds, k=k))
    
    print(f"HitRate@{k}: {np.mean(hit_rates):.4f}, NDCG@{k}: {np.mean(ndcgs):.4f}")

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