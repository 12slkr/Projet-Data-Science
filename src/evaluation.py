import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)


def compute_metrics(name, y_true, y_pred, y_proba):
    return {
        'Modele': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC_AUC': round(roc_auc_score(y_true, y_proba), 4),
        'PR_AUC': round(average_precision_score(y_true, y_proba), 4),
    }


def plot_roc_curves(models_probas, y_true, ax=None):
    """Trace les courbes ROC superposées pour plusieurs modèles."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (name, y_proba), color in zip(models_probas, colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Taux faux positifs')
    ax.set_ylabel('Taux vrais positifs')
    ax.set_title('Courbes ROC — Comparaison des modèles')
    ax.legend(loc='lower right')
    return ax


def plot_pr_curves(models_probas, y_true, ax=None):
    """Trace les courbes Precision-Recall superposées."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (name, y_proba), color in zip(models_probas, colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, label=f"{name} (AUC={pr_auc:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Courbes Precision-Recall')
    ax.legend(loc='upper right')
    return ax


def plot_confusion_matrices(models_preds, y_true):
    """Affiche les matrices de confusion en grille 2x2."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, (name, y_pred) in enumerate(models_preds):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].set_title(name)
    plt.tight_layout()
    return fig


def plot_feature_importance(fi_df, top_n=15, ax=None):
    """Bar chart horizontal des features les plus importantes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    data = fi_df.head(top_n).sort_values('Importance')
    ax.barh(data['Feature'], data['Importance'], color='#2ca02c')
    ax.set_title(f'Top {top_n} features — Gradient Boosting')
    ax.set_xlabel('Importance')
    return ax
