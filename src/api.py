# Fichier : src/api.py
import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# 1. Initialisation de l'API
app = FastAPI(title="Credit Scoring API", description="API pour prédire le risque de défaut")

# Variables globales pour stocker le modèle et le seuil
model = None
threshold = 0.53  # Valeur par défaut de sécurité


# 2. Chargement du Modèle au démarrage
@app.on_event("startup")
def load_model():
    global model, threshold

    # Chemins relatifs (Attention à l'endroit où vous lancez la commande !)
    # Si vous lancez depuis la racine du projet :
    model_path = "model/model.pkl"
    thresh_path = "model/threshold.txt"

    try:
        # Chargement du modèle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Modèle chargé avec succès.")

        # Chargement du seuil
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                threshold = float(f.read())
            print(f"✅ Seuil métier chargé : {threshold}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")


# 3. Définition du format des données attendues (Validation)
# Pour faire simple, on accepte un dictionnaire brut (les features du client)
class ClientData(BaseModel):
    features: dict


# 4. Route de Test (Pour vérifier que l'API est en vie)
@app.get("/")
def read_root():
    return {"message": "API de Credit Scoring en ligne ! 🚀"}


# 5. Route de Prédiction (Le cœur du réacteur)
@app.post("/predict")
def predict(data: ClientData):
    if not model:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    try:
        # Conversion du JSON reçu en DataFrame (1 seule ligne)
        df = pd.DataFrame([data.features])

        # Prédiction de la probabilité (Classe 1 = Défaut)
        proba = model.predict_proba(df)[:, 1][0]

        # Décision basée sur le seuil métier
        decision = "REFUSÉ" if proba >= threshold else "ACCORDÉ"

        return {
            "probability": float(proba),
            "threshold": float(threshold),
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Lancement local (si on exécute le fichier directement)
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)