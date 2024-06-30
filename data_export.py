import pandas as pd
from google.cloud import firestore

def reformat_symptom(symptom):
    if isinstance(symptom, str):
        return symptom.replace('_', ' ').strip()
    return symptom

def export_data_to_firestore(file_path, db):
    # Charger les données
    df = pd.read_csv(file_path)

    # Reformater les symptômes
    symptom_columns = [col for col in df.columns if 'Symptom_' in col]
    for col in symptom_columns:
        df[col] = df[col].apply(reformat_symptom)

    # Supprimer les doublons
    df_cleaned = df.drop_duplicates()

    # Encoder les symptômes
    symptoms_df = df_cleaned[symptom_columns].apply(lambda x: ', '.join(x.dropna()), axis=1).str.get_dummies(sep=', ')
    symptoms_df['Disease'] = df_cleaned['Disease']

    # Convertir en DataFrame sparse, en excluant la colonne 'Disease'
    sparse_symptoms_df = symptoms_df.drop(columns=['Disease']).astype(pd.SparseDtype("float", fill_value=0))
    sparse_symptoms_df['Disease'] = df_cleaned['Disease'].values

    # Convertir les DataFrames en dictionnaires pour Firestore
    diseases_and_symptoms_data = sparse_symptoms_df.to_dict(orient='records')
    symptoms_data = list(symptoms_df.columns.drop('Disease'))

    # Insérer les données dans la collection 'DiseasesAndSymptoms'
    diseases_and_symptoms_collection = db.collection('DiseasesAndSymptoms')
    for record in diseases_and_symptoms_data:
        diseases_and_symptoms_collection.add(record)

    # Insérer les symptômes dans la collection 'Symptoms'
    symptoms_collection = db.collection('Symptoms')
    for symptom in symptoms_data:
        symptoms_collection.add({'Symptom': symptom})

    return "Les données ont été insérées avec succès dans Firestore."
