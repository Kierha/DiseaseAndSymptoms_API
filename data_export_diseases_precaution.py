import pandas as pd
from google.cloud import firestore

def reformat_precaution(precaution):
    if isinstance(precaution, str):
        return precaution.strip()
    return precaution

def export_data_to_firestore_precautions(file_path, db):
    # Charger les données
    df = pd.read_csv(file_path)

    # Reformater les précautions
    precaution_columns = [col for col in df.columns if 'Precaution_' in col]
    for col in precaution_columns:
        df[col] = df[col].apply(reformat_precaution)

    # Supprimer les doublons
    df_cleaned = df.drop_duplicates()

    # Convertir en DataFrame sparse, en excluant la colonne 'Disease'
    precautions_df = df_cleaned[precaution_columns].apply(lambda x: ', '.join(x.dropna()), axis=1)
    precautions_df = precautions_df.str.get_dummies(sep=', ')
    precautions_df['Disease'] = df_cleaned['Disease']

    # Convertir les DataFrames en dictionnaires pour Firestore
    diseases_and_precautions_data = precautions_df.to_dict(orient='records')
    precautions_data = list(precautions_df.columns.drop('Disease'))

    # Insérer les données dans la collection 'DiseasesPrecaution'
    diseases_and_precautions_collection = db.collection('DiseasesPrecaution')
    for record in diseases_and_precautions_data:
        diseases_and_precautions_collection.add(record)

    # Insérer les précautions dans la collection 'Precautions'
    precautions_collection = db.collection('Precautions')
    for precaution in precautions_data:
        precautions_collection.add({'Precaution': precaution})

    return "Les données ont été insérées avec succès dans Firestore."
