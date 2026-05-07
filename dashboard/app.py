import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import shap
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Permettre l'import de src/ depuis le dossier parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ── Chargement des modèles (une seule fois, mis en cache) ─────────────────────

@st.cache_resource
def load_models():
    best_model = joblib.load("models/best_model.pkl")
    lr_model   = joblib.load("models/lr_model.pkl")
    rf_model   = joblib.load("models/rf_model.pkl")
    prep_mlp   = joblib.load("models/mlp_preprocessor.pkl")
    mlp_model  = tf.keras.models.load_model("models/mlp_model.keras")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    return best_model, lr_model, rf_model, prep_mlp, mlp_model, meta


@st.cache_data
def load_results():
    metrics = pd.read_csv("models/metrics.csv", index_col=0)
    fi      = pd.read_csv("models/feature_importance.csv")
    return metrics, fi


@st.cache_data
def load_data_with_predictions(_best_model):
    from src.preprocessing import load_data, get_features_target
    df = load_data("data/customer_churn_business_dataset.csv")
    X, y = get_features_target(df)
    df['churn_proba']    = _best_model.predict_proba(X)[:, 1]
    df['churn_predicted'] = (df['churn_proba'] >= 0.35).astype(int)
    df['revenue_at_risk'] = df['total_revenue'] * df['churn_proba']
    return df, X, y


@st.cache_data
def compute_test_curves(_lr, _rf, _gb, _prep_mlp, _mlp):
    from src.preprocessing import load_data, get_features_target, split_data
    _df = load_data("data/customer_churn_business_dataset.csv")
    _X, _y = get_features_target(_df)
    _, X_test, _, y_test = split_data(_X, _y)
    X_mlp = _prep_mlp.transform(X_test)
    probas = {
        'Logistic Regression': _lr.predict_proba(X_test)[:, 1],
        'Random Forest':       _rf.predict_proba(X_test)[:, 1],
        'Gradient Boosting':   _gb.predict_proba(X_test)[:, 1],
        'MLP Deep Learning':   _mlp.predict(X_mlp, verbose=0).ravel(),
    }
    preds = {
        'Logistic Regression': _lr.predict(X_test),
        'Random Forest':       _rf.predict(X_test),
        'Gradient Boosting':   _gb.predict(X_test),
        'MLP Deep Learning':   (_mlp.predict(X_mlp, verbose=0).ravel() >= 0.35).astype(int),
    }
    return probas, preds, y_test.to_numpy()


@st.cache_data
def compute_shap_values(_best_model, _X_full, cat_cols, num_cols):
    sample    = _X_full.sample(300, random_state=42)
    explainer = shap.TreeExplainer(_best_model.named_steps['clf'])
    X_prep    = _best_model.named_steps['prep'].transform(sample)
    sv        = explainer.shap_values(X_prep)
    ohe_names = (_best_model.named_steps['prep']
                 .named_transformers_['cat']
                 .get_feature_names_out(cat_cols).tolist())
    return sv, X_prep, num_cols + ohe_names


best_model, lr_model, rf_model, prep_mlp, mlp_model, meta = load_models()
df_metrics, fi_df = load_results()
df, X_full, y_full = load_data_with_predictions(best_model)

# ── Navigation ────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📈 Vue d'ensemble",
    "🔍 Analyse Exploratoire",
    "🤖 Comparaison Modèles",
    "🧠 Interprétabilité",
    "👤 Prédiction Individuelle",
    "🎮 Simulation Scénario",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Modèle actif : {meta['best_model']}\n"
    f"ROC-AUC : {meta['roc_auc']:.3f} | F1 : {meta['f1']:.3f}"
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📈 Vue d'ensemble":
    st.title("📈 Vue d'ensemble — Risque de Churn")

    at_risk = df[df['churn_predicted'] == 1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients à risque", f"{len(at_risk):,}", f"{len(at_risk)/len(df)*100:.1f}%")
    col2.metric("Revenu total à risque", f"{at_risk['revenue_at_risk'].sum():,.0f} €")
    col3.metric("Taux de churn prédit", f"{df['churn_predicted'].mean()*100:.1f}%")
    col4.metric("Probabilité moyenne", f"{df['churn_proba'].mean()*100:.1f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_hist = px.histogram(
            df, x='churn_proba', color='churn_predicted',
            nbins=50, barmode='overlay',
            title="Distribution des probabilités de churn",
            labels={'churn_proba': 'Probabilité de churn', 'churn_predicted': 'Classe prédite'},
            color_discrete_map={0: '#2196F3', 1: '#F44336'},
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        risk_seg = df.groupby('customer_segment')['churn_proba'].mean().reset_index()
        risk_seg.columns = ['Segment', 'Risque moyen']
        fig_seg = px.bar(
            risk_seg, x='Segment', y='Risque moyen',
            title="Risque moyen par segment client",
            color='Risque moyen', color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Top 10 clients à risque le plus élevé")
    top10 = (
        df.sort_values('churn_proba', ascending=False)
        [['customer_id', 'customer_segment', 'contract_type',
          'churn_proba', 'total_revenue', 'revenue_at_risk']]
        .head(10)
        .rename(columns={
            'customer_id': 'ID', 'customer_segment': 'Segment',
            'contract_type': 'Contrat', 'churn_proba': 'Proba Churn',
            'total_revenue': 'Revenu Total', 'revenue_at_risk': 'Revenu à Risque'
        })
    )
    top10['Proba Churn'] = top10['Proba Churn'].map('{:.1%}'.format)
    top10['Revenu à Risque'] = top10['Revenu à Risque'].map('{:,.0f} €'.format)
    st.dataframe(top10, use_container_width=True)

    rev_contract = df.groupby('contract_type')['revenue_at_risk'].sum().reset_index()
    fig_pie = px.pie(
        rev_contract, values='revenue_at_risk', names='contract_type',
        title="Répartition du revenu à risque par type de contrat"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE EXPLORATOIRE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Analyse Exploratoire":
    st.title("🔍 Analyse Exploratoire des Données")

    num_options = ['age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
                   'monthly_fee', 'total_revenue', 'payment_failures',
                   'support_tickets', 'csat_score', 'nps_score',
                   'email_open_rate', 'marketing_click_rate']
    selected_num = st.selectbox("Variable numérique à explorer", num_options)
    fig_num = px.histogram(
        df, x=selected_num, color='churn',
        barmode='overlay', nbins=40, opacity=0.7,
        title=f"Distribution de {selected_num} par churn",
        color_discrete_map={0: '#2196F3', 1: '#F44336'}
    )
    st.plotly_chart(fig_num, use_container_width=True)

    st.subheader("Taux de churn par variable catégorielle")
    cat_vars = ['contract_type', 'customer_segment', 'payment_method', 'survey_response']
    cols = st.columns(2)
    for i, var in enumerate(cat_vars):
        churn_rate = df.groupby(var)['churn'].mean().reset_index()
        churn_rate.columns = [var, 'Taux de churn']
        fig_cat = px.bar(
            churn_rate, x=var, y='Taux de churn',
            title=f"Taux de churn — {var}",
            color='Taux de churn', color_continuous_scale='Reds'
        )
        cols[i % 2].plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Heatmap de corrélation")
    num_cols_corr = ['age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
                     'avg_session_time', 'features_used', 'monthly_fee', 'total_revenue',
                     'payment_failures', 'support_tickets', 'csat_score', 'nps_score',
                     'email_open_rate', 'marketing_click_rate', 'churn']
    existing_cols = [c for c in num_cols_corr if c in df.columns]
    corr_matrix = df[existing_cols].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto='.2f', aspect='auto',
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title="Matrice de corrélation"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Boxplots comparatifs selon le churn")
    box_vars = ['payment_failures', 'nps_score', 'support_tickets', 'csat_score']
    cols2 = st.columns(2)
    for i, var in enumerate(box_vars):
        if var in df.columns:
            fig_box = px.box(
                df, x='churn', y=var,
                title=f"{var} selon le churn",
                color='churn',
                color_discrete_map={0: '#2196F3', 1: '#F44336'}
            )
            cols2[i % 2].plotly_chart(fig_box, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON DES 4 MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Comparaison Modèles":
    st.title("🤖 Comparaison des 4 Modèles")

    st.subheader("Tableau de métriques comparatif")
    styled = df_metrics.style.highlight_max(axis=0, color='#c6efce').format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    models_proba, models_pred, y_test = compute_test_curves(
        lr_model, rf_model, best_model, prep_mlp, mlp_model)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Courbes ROC")
        fig_roc = go.Figure()
        for (name, y_proba), color in zip(models_proba.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc_val:.3f})",
                line=dict(color=color)
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     line=dict(dash='dash', color='grey'), showlegend=False))
        fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("Courbes Precision-Recall")
        fig_pr = go.Figure()
        for (name, y_proba), color in zip(models_proba.items(), colors):
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc_val = auc(recall, precision)
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision, mode='lines',
                name=f"{name} (AUC={pr_auc_val:.3f})",
                line=dict(color=color)
            ))
        fig_pr.update_layout(xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("Matrices de confusion")
    cm_cols = st.columns(4)
    for col, (name, y_pred) in zip(cm_cols, models_pred.items()):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            labels=dict(x='Prédit', y='Réel'),
            x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
            title=name
        )
        col.plotly_chart(fig_cm, use_container_width=True)

    st.info(
        "**Conclusion :** Le Gradient Boosting obtient le meilleur compromis F1 / ROC-AUC. "
        "Il gère nativement le déséquilibre de classes via `class_weight='balanced'` et ses "
        "arbres peu profonds (max_depth=4) limitent l'overfitting. Il est choisi comme "
        "modèle final pour le dashboard."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERPRÉTABILITÉ
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Interprétabilité":
    st.title("🧠 Interprétabilité du Modèle")

    st.subheader("Top 15 features — Gradient Boosting")
    top15 = fi_df.head(15).sort_values('Importance')
    fig_fi = px.bar(
        top15, x='Importance', y='Feature',
        orientation='h', color='Importance',
        color_continuous_scale='Greens',
        title="Importance des features (Gradient Boosting)"
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("SHAP Summary Plot — échantillon de 300 clients")

    with st.spinner("Calcul des valeurs SHAP en cours..."):
        shap_values, X_prep, feat_names = compute_shap_values(
            best_model, X_full,
            meta['cat_cols'], meta['num_cols']
        )

    import matplotlib.pyplot as plt
    fig_shap, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_prep, feature_names=feat_names, show=False, max_display=15)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Interprétation métier des variables clés")
    interpretations = {
        "nps_score": "Un NPS faible ou négatif est un signal fort de départ imminent. Prioriser les détracteurs.",
        "csat_score": "La satisfaction client est directement liée à la rétention. Scores < 3 = risque élevé.",
        "tenure_months": "Les clients récents churnent plus. Renforcer l'onboarding les 6 premiers mois.",
        "payment_failures": "Tout échec de paiement non résolu multiplie le risque de churn par 3.",
        "support_tickets": "Accumulation de tickets = friction produit. Indicateur d'insatisfaction latente.",
        "contract_type": "Les contrats mensuels ont 4x plus de churn que les contrats annuels.",
        "usage_growth_rate": "Une croissance d'usage négative anticipe le désengagement.",
    }
    for var, text in interpretations.items():
        st.markdown(f"- **{var}** : {text}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRÉDICTION INDIVIDUELLE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👤 Prédiction Individuelle":
    st.title("👤 Prédiction de Churn — Client Individuel")

    with st.form("prediction_form"):
        st.subheader("Informations client")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender          = st.selectbox("Genre", ["Male", "Female"])
            age             = st.slider("Âge", 18, 80, 35)
            customer_seg    = st.selectbox("Segment", ["Individual", "SME", "Enterprise"])
            tenure_months   = st.slider("Ancienneté (mois)", 1, 120, 24)
            signup_channel  = st.selectbox("Canal inscription", ["Web", "Mobile", "Partner"])
            contract_type   = st.selectbox("Type de contrat", ["Monthly", "Yearly"])
            payment_method  = st.selectbox("Méthode de paiement", ["Card", "PayPal", "Bank Transfer"])

        with col2:
            monthly_logins      = st.slider("Connexions/mois", 0, 60, 15)
            weekly_active_days  = st.slider("Jours actifs/semaine", 0, 7, 4)
            avg_session_time    = st.slider("Durée session moy. (min)", 0.0, 120.0, 20.0)
            features_used       = st.slider("Features utilisées", 0, 20, 8)
            usage_growth_rate   = st.slider("Taux croissance usage", -1.0, 2.0, 0.1)
            last_login_days_ago = st.slider("Jours depuis dernière co.", 0, 180, 10)
            monthly_fee         = st.slider("Frais mensuels (€)", 10, 500, 80)

        with col3:
            total_revenue           = st.slider("Revenu total (€)", 100, 50000, 3000)
            payment_failures        = st.slider("Échecs de paiement", 0, 10, 0)
            discount_applied        = st.selectbox("Remise appliquée", ["No", "Yes"])
            price_increase_last_3m  = st.selectbox("Hausse prix récente", ["No", "Yes"])
            support_tickets         = st.slider("Tickets support", 0, 20, 1)
            avg_resolution_time     = st.slider("Temps résolution moy. (h)", 0.0, 72.0, 12.0)
            complaint_type          = st.selectbox("Type de plainte", ["Unknown", "Service", "Billing", "Technical"])
            csat_score              = st.slider("Score satisfaction (1-5)", 1.0, 5.0, 3.5)
            escalations             = st.slider("Escalades", 0, 10, 0)
            email_open_rate         = st.slider("Taux ouverture emails", 0.0, 1.0, 0.3)
            marketing_click_rate    = st.slider("Taux clic marketing", 0.0, 1.0, 0.1)
            nps_score               = st.slider("NPS Score", -100, 100, 20)
            survey_response         = st.selectbox("Réponse enquête", ["Satisfied", "Neutral", "Dissatisfied"])
            referral_count          = st.slider("Parrainages", 0, 20, 1)

        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        disc_val = 1 if discount_applied == "Yes" else 0
        price_val = 1 if price_increase_last_3m == "Yes" else 0

        input_data = pd.DataFrame([{
            'gender': gender, 'age': age, 'customer_segment': customer_seg,
            'tenure_months': tenure_months, 'signup_channel': signup_channel,
            'contract_type': contract_type, 'monthly_logins': monthly_logins,
            'weekly_active_days': weekly_active_days, 'avg_session_time': avg_session_time,
            'features_used': features_used, 'usage_growth_rate': usage_growth_rate,
            'last_login_days_ago': last_login_days_ago, 'monthly_fee': monthly_fee,
            'total_revenue': total_revenue, 'payment_method': payment_method,
            'payment_failures': payment_failures, 'discount_applied': disc_val,
            'price_increase_last_3m': price_val, 'support_tickets': support_tickets,
            'avg_resolution_time': avg_resolution_time, 'complaint_type': complaint_type,
            'csat_score': csat_score, 'escalations': escalations,
            'email_open_rate': email_open_rate, 'marketing_click_rate': marketing_click_rate,
            'nps_score': nps_score, 'survey_response': survey_response,
            'referral_count': referral_count,
            'tickets_per_tenure': support_tickets / (tenure_months + 1),
            'avg_monthly_revenue': total_revenue / (tenure_months + 1),
            'financial_risk_flag': int(payment_failures > 0 and price_val == 1),
        }])

        proba = best_model.predict_proba(input_data)[0, 1]
        revenue_at_risk = total_revenue * proba

        if proba < 0.3:
            risk_label, risk_color = "Faible", "green"
        elif proba < 0.6:
            risk_label, risk_color = "Modéré", "orange"
        else:
            risk_label, risk_color = "Élevé", "red"

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': '%'},
                title={'text': f"Risque de Churn — {risk_label}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': '#c8e6c9'},
                        {'range': [30, 60], 'color': '#fff9c4'},
                        {'range': [60, 100], 'color': '#ffcdd2'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 4}, 'value': proba * 100}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res2:
            st.metric("Probabilité de churn", f"{proba:.1%}")
            st.metric("Revenu estimé à risque", f"{revenue_at_risk:,.0f} €")
            st.markdown(f"**Niveau de risque : :{risk_color}[{risk_label}]**")

        st.subheader("Explication SHAP — cette prédiction")
        try:
            import matplotlib.pyplot as plt
            explainer = shap.TreeExplainer(best_model.named_steps['clf'])
            X_prep_ind = best_model.named_steps['prep'].transform(input_data)
            sv = explainer(X_prep_ind)
            ohe_names = (
                best_model.named_steps['prep']
                .named_transformers_['cat']
                .get_feature_names_out(meta['cat_cols']).tolist()
            )
            sv.feature_names = meta['num_cols'] + ohe_names
            fig_wf, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(sv[0], max_display=12, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.warning(f"SHAP waterfall non disponible : {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SIMULATION DE SCÉNARIO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎮 Simulation Scénario":
    st.title("🎮 Simulation de Scénario — Impact en Temps Réel")

    st.markdown("Modifiez les variables clés et observez l'impact immédiat sur le risque de churn.")

    base = df.median(numeric_only=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Variables impactantes")
        payment_failures_sim = st.slider(
            "Échecs de paiement", 0, 10, int(base.get('payment_failures', 0)))
        nps_score_sim = st.slider(
            "NPS Score", -100, 100, int(base.get('nps_score', 20)))
        support_tickets_sim = st.slider(
            "Tickets support", 0, 20, int(base.get('support_tickets', 1)))
        usage_growth_sim = st.slider(
            "Taux croissance usage", -1.0, 2.0,
            float(base.get('usage_growth_rate', 0.1)), step=0.05)
        tenure_months_sim = st.slider(
            "Ancienneté (mois)", 1, 120, int(base.get('tenure_months', 24)))

    sim_profile = {
        'gender': 'Male', 'age': int(base.get('age', 35)),
        'customer_segment': 'Individual', 'signup_channel': 'Web',
        'contract_type': 'Monthly', 'payment_method': 'Card',
        'monthly_logins': int(base.get('monthly_logins', 15)),
        'weekly_active_days': int(base.get('weekly_active_days', 4)),
        'avg_session_time': float(base.get('avg_session_time', 20.0)),
        'features_used': int(base.get('features_used', 8)),
        'last_login_days_ago': int(base.get('last_login_days_ago', 10)),
        'monthly_fee': int(base.get('monthly_fee', 80)),
        'total_revenue': int(base.get('total_revenue', 3000)),
        'discount_applied': 0, 'price_increase_last_3m': 0,
        'avg_resolution_time': float(base.get('avg_resolution_time', 12.0)),
        'complaint_type': 'Unknown',
        'csat_score': float(base.get('csat_score', 3.5)),
        'escalations': int(base.get('escalations', 0)),
        'email_open_rate': float(base.get('email_open_rate', 0.3)),
        'marketing_click_rate': float(base.get('marketing_click_rate', 0.1)),
        'survey_response': 'Neutral',
        'referral_count': int(base.get('referral_count', 1)),
        'payment_failures': payment_failures_sim,
        'nps_score': nps_score_sim,
        'support_tickets': support_tickets_sim,
        'usage_growth_rate': usage_growth_sim,
        'tenure_months': tenure_months_sim,
    }
    sim_profile['tickets_per_tenure']  = sim_profile['support_tickets'] / (sim_profile['tenure_months'] + 1)
    sim_profile['avg_monthly_revenue'] = sim_profile['total_revenue'] / (sim_profile['tenure_months'] + 1)
    sim_profile['financial_risk_flag'] = int(
        sim_profile['payment_failures'] > 0 and sim_profile['price_increase_last_3m'] == 1)

    sim_df    = pd.DataFrame([sim_profile])
    proba_sim = best_model.predict_proba(sim_df)[0, 1]

    if proba_sim < 0.3:
        risk_sim, color_sim = "Faible", "green"
    elif proba_sim < 0.6:
        risk_sim, color_sim = "Modéré", "orange"
    else:
        risk_sim, color_sim = "Élevé", "red"

    with col2:
        fig_gauge_sim = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba_sim * 100,
            number={'suffix': '%'},
            title={'text': f"Probabilité de Churn — {risk_sim}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color_sim},
                'steps': [
                    {'range': [0, 30], 'color': '#c8e6c9'},
                    {'range': [30, 60], 'color': '#fff9c4'},
                    {'range': [60, 100], 'color': '#ffcdd2'},
                ],
            }
        ))
        st.plotly_chart(fig_gauge_sim, use_container_width=True)

        impacts = {}
        for var, val in [
            ('payment_failures', payment_failures_sim),
            ('nps_score', nps_score_sim),
            ('support_tickets', support_tickets_sim),
            ('usage_growth_rate', usage_growth_sim),
            ('tenure_months', tenure_months_sim),
        ]:
            tmp = sim_df.copy()
            tmp[var] = base.get(var, val)
            p = best_model.predict_proba(tmp)[0, 1]
            impacts[var] = proba_sim - p

        impact_df = pd.DataFrame(list(impacts.items()), columns=['Variable', 'Impact'])
        impact_df['Couleur'] = impact_df['Impact'].apply(
            lambda x: 'Augmente risque' if x > 0 else 'Réduit risque')
        fig_impact = px.bar(
            impact_df, x='Impact', y='Variable', orientation='h',
            color='Couleur',
            color_discrete_map={'Augmente risque': '#F44336', 'Réduit risque': '#4CAF50'},
            title="Impact de chaque variable sur le score de risque"
        )
        st.plotly_chart(fig_impact, use_container_width=True)

    st.subheader("Action recommandée")
    if proba_sim < 0.3:
        st.success("Risque faible. Ce client est stable. Maintenir la qualité de service.")
    elif proba_sim < 0.6:
        st.warning(
            "Risque modéré. Recommandations : contacter le client de manière proactive, "
            "proposer une offre de fidélisation, résoudre les tickets en attente."
        )
    else:
        st.error(
            "Risque élevé — intervention urgente requise. "
            "Actions prioritaires : appel commercial immédiat, offre de remise personnalisée, "
            "escalade vers le service client dédié, révision du contrat."
        )
