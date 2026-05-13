import joblib
import numpy as np
import pandas as pd
import math
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

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

print("\n--- Splitting by users (80/20) ---")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

print(f"Train rows: {len(X_train)}, users: {train_groups.nunique()}")
print(f"Test rows: {len(X_test)}, users: {test_groups.nunique()}")
print(f"Positive train: {(y_train == 1).sum()}, Negative train: {(y_train == 0).sum()}")
print(f"Positive test: {(y_test == 1).sum()}, Negative test: {(y_test == 0).sum()}")

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

print("\n--- Cross-validation (GroupKFold by user_id) ---")
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
        print(f"ROC-AUC on folds: {cv_scores}")
        print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
else:
    print("Cross-validation skipped: too few users")

pipeline.fit(X_train, y_train)

y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_train_pred_proba)
print(f"\n--- In-sample metric (for sanity only) ---")
print(f"Train ROC-AUC: {train_auc:.4f}")

print("\n" + "=" * 70)
print("EVALUATION ON HELD-OUT USERS")
print("=" * 70)


y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
test_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"Test ROC-AUC: {test_auc:.4f}")

def hitrate_at_k(y_true, y_pred_proba, user_ids, k=10):
    """
    HitRate@K: доля пользователей, у которых хотя бы один релевантный объект
    попал в топ-K рекомендаций.
    """
    df_pred = pd.DataFrame({
        'user_id': user_ids,
        'label': y_true,
        'proba': y_pred_proba
    })
    
    hits = 0
    total_users = df_pred['user_id'].nunique()
    
    for user_id, group in df_pred.groupby('user_id'):
        top_k = group.nlargest(k, 'proba')['label'].values
        if top_k.sum() > 0:
            hits += 1
    
    return hits / total_users if total_users > 0 else 0.0

def ndcg_at_k(y_true, y_pred_proba, user_ids, k=10):
    """
    NDCG@K: нормализованный дисконтированный кумулятивный выигрыш.
    Учитывает позиции всех релевантных объектов.
    """
    df_pred = pd.DataFrame({
        'user_id': user_ids,
        'label': y_true,
        'proba': y_pred_proba
    })
    
    total_ndcg = 0.0
    total_users = 0
    
    for user_id, group in df_pred.groupby('user_id'):
        total_users += 1
        
        sorted_group = group.sort_values('proba', ascending=False).head(k)
        
        dcg = 0.0
        for i, (idx, row) in enumerate(sorted_group.iterrows(), start=1):
            rel = row['label']
            dcg += rel / math.log2(i + 1)
        
        ideal_relevances = sorted(group['label'].values, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances, start=1):
            idcg += rel / math.log2(i + 1)
        
        if idcg > 0:
            total_ndcg += dcg / idcg
    
    return total_ndcg / total_users if total_users > 0 else 0.0

print("\n--- Ranking Metrics on Held-Out Users ---")
ranking_results = []
for k in [1, 3, 5, 10]:
    hr = hitrate_at_k(y_test, y_test_pred_proba, test_groups, k=k)
    ndcg = ndcg_at_k(y_test, y_test_pred_proba, test_groups, k=k)
    ranking_results.append({'k': k, 'hitrate': hr, 'ndcg': ndcg})
    print(f"K={k:2d} | HitRate={hr:.4f} | NDCG={ndcg:.4f}")

results_path = "evaluation_results.csv"
file_exists = False
try:
    with open(results_path, 'r') as f:
        file_exists = True
except FileNotFoundError:
    pass

with open(results_path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['timestamp', 'train_auc', 'test_auc', 'k', 'hitrate', 'ndcg'])
    
    for res in ranking_results:
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(train_auc, 4),
            round(test_auc, 4),
            res['k'],
            round(res['hitrate'], 4),
            round(res['ndcg'], 4)
        ])

print(f"\nResults saved to {results_path}")

joblib.dump(pipeline.named_steps["model"], MODEL_PATH)
joblib.dump(pipeline.named_steps["scaler"], SCALER_PATH)
joblib.dump(feature_names, FEATURE_NAMES_PATH)
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_names": feature_names,
    },
    MODEL_BUNDLE_PATH,
)

print("\nModel, scaler, feature_names and model bundle saved successfully.")

coefficients = pipeline.named_steps["model"].coef_[0]
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

print("\n Training and evaluation completed!")