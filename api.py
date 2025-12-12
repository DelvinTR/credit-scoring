# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

# 1. Chargement du modèle (Adaptez le chemin si besoin)
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


# 2. Définition des données attendues (Ignore les colonnes en trop)
class ClientData(BaseModel):
    features: dict

    class Config:
        extra = "allow"


# 3. La fonction de prédiction
@app.post("/predict")
def predict(data: ClientData):
    # Transformation en DataFrame
    df = pd.DataFrame([data.features])

    # Alignement des colonnes (Astuce pour éviter l'erreur de shape)
    try:
        expected_cols = model.booster_.feature_name()
    except:
        expected_cols = model.feature_name_  # Si sklearn

    df = df.reindex(columns=expected_cols, fill_value=0)

    # Prédiction
    proba = model.predict_proba(df)[:, 1][0]
    decision = "REFUSÉ" if proba >= 0.53 else "ACCORDÉ"  # Mettez votre seuil

    return {"probability": float(proba), "decision": decision, "threshold": 0.53}


# 4. L'alias pour le cahier des charges
@app.post("/invocations")
def invocations(data: ClientData):
    return predict(data)