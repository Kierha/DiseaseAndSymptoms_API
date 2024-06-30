import os
import shutil
from typing import Dict, List
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Importer les fonctions d'exportation et de KNN
from data_export import export_data_to_firestore
from data_export_diseases_precaution import export_data_to_firestore_precautions
from model_knn import JaccardKNN, train_knn_model, predict_disease, encode_symptoms, initialize_model

# Initialiser l'application FastAPI
app = FastAPI()

# Chemin vers ton fichier de clé de service Firestore
SERVICE_ACCOUNT_KEY_PATH = "env/medtrack-420ed-firebase-adminsdk-yobin-c036a6d4f5.json"

# Initialiser Firestore et entraîner le modèle au démarrage
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Connexion à Firestore établie.")
    
    knn_model, symptom_columns = initialize_model(db)
    print("Modèle KNN entraîné avec succès.")
except Exception as e:
    print(f"Erreur lors de la connexion à Firestore ou de l'entraînement du modèle: {e}")
    db = None
    knn_model = None
    symptom_columns = None

@app.get("/")
async def root():
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    return {"message": "API is up and running with Firestore connection"}

@app.get("/check-firestore")
async def check_firestore():
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    try:
        # Vérifie la connexion en récupérant un document fictif
        doc_ref = db.collection('test').document('check-connection')
        doc = doc_ref.get()
        if doc.exists:
            return {"message": "Connexion à Firestore est réussie"}
        else:
            return {"message": "Le document de vérification n'existe pas, mais la connexion est réussie"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la vérification de la connexion à Firestore: {e}")

@app.get("/temperature", response_model=List[Dict])
async def get_temperature():
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    try:
        temperature_ref = db.collection('Temperature')
        docs = temperature_ref.stream()
        temperature_data = []
        for doc in docs:
            temperature_data.append(doc.to_dict())
        return temperature_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des documents de la collection Temperature: {e}")

@app.get("/weight", response_model=List[Dict])
async def get_weight():
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    try:
        weight_ref = db.collection('Weight')
        docs = weight_ref.stream()
        weight_data = []
        for doc in docs:
            weight_data.append(doc.to_dict())
        return weight_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des documents de la collection Weight: {e}")

@app.post("/export-data/")
async def export_data(file: UploadFile = File(...)):
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    try:
        # Créer un répertoire temporaire spécifique à l'application
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Chemin complet du fichier temporaire
        file_path = os.path.join(temp_dir, file.filename)
        
        # Sauvegarder le fichier temporairement
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Appeler la fonction d'exportation
        message = export_data_to_firestore(file_path, db)
        
        # Supprimer le fichier temporaire après traitement
        os.remove(file_path)

        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-data-precautions/")
async def export_data_precautions(file: UploadFile = File(...)):
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    try:
        # Créer un répertoire temporaire spécifique à l'application
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Chemin complet du fichier temporaire
        file_path = os.path.join(temp_dir, file.filename)
        
        # Sauvegarder le fichier temporairement
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Appeler la fonction d'exportation
        message = export_data_to_firestore_precautions(file_path, db)
        
        # Supprimer le fichier temporaire après traitement
        os.remove(file_path)

        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SymptomRequest(BaseModel):
    symptoms: str

class PredictionResponse(BaseModel):
    predictions: List[str]

@app.post("/predict-disease/", response_model=PredictionResponse)
async def predict_disease_endpoint(symptom_request: SymptomRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Connexion à Firestore échouée")
    if knn_model is None or symptom_columns is None:
        raise HTTPException(status_code=500, detail="Modèle KNN non entraîné")
    try:
        diseases_prediction = predict_disease(knn_model, symptom_columns, symptom_request.symptoms)
        return {"predictions": diseases_prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exécuter l'application avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
