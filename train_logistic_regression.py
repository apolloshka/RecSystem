import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from src.db.clickhouse_client import get_client

print("=== Training Robust Logistic Regression ===")

client = get_client()

# 1. Загрузка данных
result = client.query("SELECT * FROM ml_dataset")
df = pd.DataFrame(result.result_rows)

# 2. Разделение на признаки (X) и целевую переменную (y)
X = df.iloc[:, 3:-1]
y = df.iloc[:, 2]

print(f"Dataset: {len(df)} rows, {X.shape[1]} features")
print(f"Positive: {(y == 1).sum()}, Negative: {(y == 0).sum()}")

# 3. --- УЛУЧШЕНИЕ 1: Стратифицированный сплит ---
# Параметр 'stratify=y' гарантирует, что пропорции классов в train и test выборках будут такими же, как в исходном датасете.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # <-- ЭТО ВАЖНО!
)

# 4. Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. --- УЛУЧШЕНИЕ 2: Борьба с дисбалансом классов ---
# Параметр 'class_weight="balanced"' автоматически назначает веса классам обратно пропорционально их частоте.
# Это заставит модель "обращать больше внимания" на редкий класс.
model = LogisticRegression(
    C=0.001,                # Регуляризация (1.0 — неплохое значение по умолчанию)
    max_iter=1000, 
    random_state=42, 
    class_weight="balanced"  # <-- ЭТО ВАЖНО!
)

# 6. --- УЛУЧШЕНИЕ 3: Кросс-валидация для надежной оценки ---
# Используем стратифицированные k-fold блоки, чтобы каждая складка (fold) также была сбалансирована по классам.
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy, scoring='roc_auc')

print(f"\n--- Результаты кросс-валидации (на обучающей выборке) ---")
print(f"ROC-AUC на каждой складке: {cv_scores}")
print(f"Средний ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. Обучение финальной модели на ВСЕХ обучающих данных
model.fit(X_train_scaled, y_train)

# 8. Оценка на тестовой выборке
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

test_auc = roc_auc_score(y_test, y_pred_proba)
test_f1 = f1_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)

print(f"\n--- Финальная оценка на тестовой выборке ---")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# 9. Сохранение модели и скейлера
joblib.dump(model, 'recommendation_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("\nModel and scaler saved successfully.")

# 10. Важность признаков (коэффициенты)
feature_names = X.columns.tolist() # Теперь названия признаков не потеряются!
coefficients = model.coef_[0]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_importance': np.abs(coefficients)
}).sort_values('abs_importance', ascending=False)

print("\nFeature importance (by coefficient magnitude):")
for i, row in importance_df.iterrows():
    # Большое положительное значение -> увеличивает шанс на positive
    # Большое отрицательное значение -> увеличивает шанс на negative
    print(f"  {row['feature']}: {row['coefficient']:+.4f}")