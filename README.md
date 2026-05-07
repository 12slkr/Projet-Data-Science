# Projet Data Science M2 — Rétention Client & Prédiction du Churn
## EFREI 2025-26 | Saphir LANKRI & Wacil BETTAHAR - Groupe 12

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation

### Étape 1 — Entraîner les modèles (une seule fois)
```bash
python train.py
```

### Étape 2 — Lancer le dashboard (autant de fois que voulu)
```bash
streamlit run dashboard/app.py
```

## Structure du projet
- `data/` : dataset CSV
- `src/` : modules de preprocessing et d'évaluation
- `models/` : modèles sauvegardés (.pkl, .keras)
- `dashboard/` : application Streamlit (6 pages)
- `notebooks/` : notebook d'exploration EDA
- `train.py` : script d'entraînement (exécuté une seule fois)

## Modèles entraînés
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting (modèle final)
4. MLP Deep Learning (Keras)

## Dataset
`customer_churn_business_dataset.csv` — 10 000 clients, 32 variables, déséquilibre 89.8/10.2%
