import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import (
    load_data, get_features_target,
    get_column_lists, build_preprocessor, split_data
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel('ERROR')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True)

print("=" * 55)
print("  ENTRAÎNEMENT DES MODÈLES — PROJET CHURN M2")
print("=" * 55)

# ── 1. Chargement ─────────────────────────────────────────
print("\n[1/6] Chargement et prétraitement des données...")
df = load_data("data/customer_churn_business_dataset.csv")
X, y = get_features_target(df)
num_cols, cat_cols = get_column_lists(X)
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"  Train : {X_train.shape} | Test : {X_test.shape}")
print(f"  Taux churn train : {y_train.mean():.3f} | test : {y_test.mean():.3f}")

# ── 2. Logistic Regression (baseline) ────────────────────
print("\n[2/6] Entraînement Logistic Regression (baseline)...")
preprocessor_lr = build_preprocessor(num_cols, cat_cols)
pipe_lr = Pipeline([
    ('prep', preprocessor_lr),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=1.0,
        random_state=SEED,
        solver='lbfgs'
    ))
])
pipe_lr.fit(X_train, y_train)
print("  Logistic Regression : OK")

# ── 3. Random Forest ──────────────────────────────────────
print("\n[3/6] Entraînement Random Forest...")
preprocessor_rf = build_preprocessor(num_cols, cat_cols)
pipe_rf = Pipeline([
    ('prep', preprocessor_rf),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ))
])
pipe_rf.fit(X_train, y_train)
print("  Random Forest : OK")

# ── 4. Gradient Boosting (modèle final recommandé) ────────
print("\n[4/6] Entraînement Gradient Boosting...")
preprocessor_gb = build_preprocessor(num_cols, cat_cols)
pipe_gb = Pipeline([
    ('prep', preprocessor_gb),
    ('clf', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=SEED
    ))
])
pipe_gb.fit(X_train, y_train)
print("  Gradient Boosting : OK")

# ── 5. MLP Deep Learning (Keras) ──────────────────────────
print("\n[5/6] Entraînement MLP Deep Learning...")

# Preprocessor dédié au MLP (Keras ne s'intègre pas dans sklearn.Pipeline)
prep_mlp = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
])
X_train_mlp = prep_mlp.fit_transform(X_train)
X_test_mlp = prep_mlp.transform(X_test)  # transform seulement, pas fit !

# Convertir en numpy pour éviter les erreurs d'index pandas avec Keras
y_train_np = y_train.to_numpy()
y_test_np  = y_test.to_numpy()

# Gestion du déséquilibre via class_weight
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_np)
class_weights = {0: cw[0], 1: cw[1]}

# Architecture MLP — 2 couches cachées
mlp = keras.Sequential([
    layers.Input(shape=(X_train_mlp.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
mlp.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
mlp.fit(
    X_train_mlp, y_train_np,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=0
)
print("  MLP Deep Learning : OK")

# ── 6. Évaluation comparative et sauvegarde ───────────────
print("\n[6/6] Évaluation comparative et sauvegarde...")

# Seuil de décision optimisé pour le MLP (déséquilibre 90/10)
MLP_THRESHOLD = 0.35


def evaluate(name, y_true, y_pred, y_proba):
    return {
        'Modele': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC_AUC': round(roc_auc_score(y_true, y_proba), 4),
        'PR_AUC': round(average_precision_score(y_true, y_proba), 4),
    }


y_proba_mlp = mlp.predict(X_test_mlp, verbose=0).ravel()
y_pred_mlp = (y_proba_mlp >= MLP_THRESHOLD).astype(int)

rows = [
    evaluate('Logistic Regression', y_test_np, pipe_lr.predict(X_test), pipe_lr.predict_proba(X_test)[:, 1]),
    evaluate('Random Forest',       y_test_np, pipe_rf.predict(X_test), pipe_rf.predict_proba(X_test)[:, 1]),
    evaluate('Gradient Boosting',   y_test_np, pipe_gb.predict(X_test), pipe_gb.predict_proba(X_test)[:, 1]),
    evaluate('MLP Deep Learning',   y_test_np, y_pred_mlp, y_proba_mlp),
]
df_metrics = pd.DataFrame(rows).set_index('Modele')
print("\n" + df_metrics.to_string())

# Feature importance du meilleur modèle (Gradient Boosting)
ohe_names = (
    pipe_gb.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(cat_cols).tolist()
)
feature_names = num_cols + ohe_names
importances = pipe_gb.named_steps['clf'].feature_importances_
fi_df = pd.DataFrame({
    'Feature': feature_names[:len(importances)],
    'Importance': importances
}).sort_values('Importance', ascending=False).head(20)

# Sauvegarde
joblib.dump(pipe_gb, "models/best_model.pkl")
joblib.dump(pipe_lr, "models/lr_model.pkl")
joblib.dump(pipe_rf, "models/rf_model.pkl")
joblib.dump(prep_mlp, "models/mlp_preprocessor.pkl")
mlp.save("models/mlp_model.keras")
df_metrics.to_csv("models/metrics.csv")
fi_df.to_csv("models/feature_importance.csv", index=False)

best_row = df_metrics.loc['Gradient Boosting']
meta = {
    'best_model': 'Gradient Boosting',
    'mlp_threshold': MLP_THRESHOLD,
    'churn_threshold': 0.35,
    'f1': float(best_row['F1']),
    'roc_auc': float(best_row['ROC_AUC']),
    'feature_names': feature_names,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 55)
print("  ENTRAÎNEMENT TERMINÉ — modèles sauvegardés dans models/")
print("  Lancer maintenant : streamlit run dashboard/app.py")
print("=" * 55)
