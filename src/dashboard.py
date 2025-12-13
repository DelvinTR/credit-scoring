import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

# URL de l'API (Backend)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


# =========================================================
# 1. CHARGEMENT DES DONNÉES ET DU MODÈLE (CACHE)
# =========================================================
@st.cache_data
def load_data():
    """Charge les données clients (Test set)"""
    try:
        df = pd.read_csv("data/processed/train_final.csv")
        # Nettoyage des colonnes (pour correspondre au modèle)
        df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

        # On ne garde que les 2000 premiers pour que le dashboard soit rapide
        if len(df) > 2000:
            df = df.sample(2000, random_state=42)
        return df
    except Exception as e:
        st.error(f"Erreur chargement données: {e}")
        return None


@st.cache_resource
def load_model_for_shap():
    """Charge le modèle uniquement pour l'explicabilité graphique"""
    try:
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None


# Chargement initial
df = load_data()
model = load_model_for_shap()

# =========================================================
# 2. BARRE LATÉRALE (SÉLECTION ID)
# =========================================================
st.sidebar.title("🔍 Recherche Dossier")
st.sidebar.markdown("---")

if df is not None:
    # Liste des IDs disponibles
    id_list = df["SK_ID_CURR"].unique()
    selected_id = st.sidebar.selectbox("Numéro de dossier (ID Client)", id_list)

    # Bouton d'action
    launch_analysis = st.sidebar.button("Lancer l'analyse")

# =========================================================
# 3. CORPS PRINCIPAL
# =========================================================
st.title("Dashboard d'Octroi de Crédit")

if df is not None and launch_analysis:

    # Récupération des données du client choisi
    client_row = df[df["SK_ID_CURR"] == selected_id].iloc[0]

    # Préparation pour l'API (Nettoyage)
    features = (
        client_row.drop(["SK_ID_CURR", "TARGET"], errors="ignore")
        .replace([np.inf, -np.inf, np.nan], None)
        .to_dict()
    )

    # -----------------------------------------------------
    # A. APPEL API (LE SCORE)
    # -----------------------------------------------------
    with st.spinner("Interrogation du modèle de Scoring via API..."):
        try:
            # On envoie les données à l'API
            response = requests.post(API_URL, json={"features": features})

            if response.status_code == 200:
                result = response.json()

                # Récupération des résultats
                score = result["probability"]
                threshold = result["threshold"]
                decision = result["decision"]

                # --- AFFICHAGE JAUGE ET DÉCISION ---
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Couleur dynamique
                    color_gauge = "green" if decision == "ACCORDÉ" else "red"
                    st.metric(label="Décision Recommandée", value=decision)

                    # Jauge Plotly
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=score,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "Probabilité de Défaut"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": color_gauge},
                                "threshold": {
                                    "line": {"color": "black", "width": 4},
                                    "thickness": 0.75,
                                    "value": threshold,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    st.subheader("Informations Clés")
                    st.write(f"**ID Client :** {selected_id}")
                    st.write(
                        f"**Revenu Annuel :** {client_row.get('AMT_INCOME_TOTAL', 0):,.0f} $"
                    )
                    st.write(
                        f"**Montant du Crédit :** {client_row.get('AMT_CREDIT', 0):,.0f} $"
                    )
                    st.write(
                        f"**Ancienneté Emploi :** {client_row.get('DAYS_EMPLOYED', 0):.0f} jours"
                    )

            else:
                st.error(f"Erreur API : {response.status_code}")

        except Exception as e:
            st.error(
                f"Impossible de contacter l'API. Vérifiez qu'elle est lancée. ({e})"
            )

    # -----------------------------------------------------
    # B. JUSTIFICATION (SHAP LOCAL)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Justification de la décision (Interprétabilité)")

    if model:
        with st.spinner("Calcul des facteurs d'influence (SHAP)..."):

            # 1. Préparation des données pour SHAP (DataFrame)
            X_client = pd.DataFrame([features])

            # 2. Calcul SHAP via le modèle chargé localement
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_client)

            # 3. Gestion des dimensions (Liste vs Array)
            # LightGBM binaire renvoie souvent une liste [Classe0, Classe1]
            if isinstance(shap_values, list):
                vals = shap_values[1]  # Classe 1 (Défaut)
            else:
                vals = shap_values

            # Idem pour expected_value
            base_val = explainer.expected_value
            if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
                if len(base_val) > 1:
                    base_val = base_val[1]
                else:
                    base_val = base_val[0]

            # 4. Création du graphique Waterfall
            st.write(
                "Ce graphique montre les critères qui ont le plus influencé la note de ce client précis."
            )

            # Création de l'objet Explanation
            exp = shap.Explanation(
                values=vals[0],  # Valeurs pour ce client
                base_values=base_val,  # Valeur moyenne
                data=X_client.iloc[0],  # Données brutes
                feature_names=X_client.columns,
            )

            # Affichage Matplotlib dans Streamlit
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=10, show=False)
            st.pyplot(fig_shap)

            # Petit texte explicatif
            st.info(
                """
            **Comment lire ce graphique ?**
            - **Rouge** : Critères qui augmentent le risque (Points négatifs).
            - **Bleu** : Critères qui diminuent le risque (Points positifs).
            - La taille de la barre indique l'importance du critère.
            """
            )

else:
    st.info(
        "Veuillez sélectionner un ID client dans la barre latérale et cliquer sur 'Lancer l'analyse'."
    )
