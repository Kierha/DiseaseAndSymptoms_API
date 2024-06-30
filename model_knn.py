import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, classification_report

# Classe JaccardKNN
class JaccardKNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _get_neighbors(self, x):
        distances = [jaccard_score(x, x_train, average='binary') for x_train in self.X_train]
        nearest_neighbors = np.argsort(distances)[-self.n_neighbors:]
        nearest_labels = self.y_train.iloc[nearest_neighbors]
        return nearest_labels

    def predict(self, X):
        return [self._get_neighbors(x).mode()[0] for x in X]

    def predict_proba(self, X):
        proba_predictions = []
        for x in X:
            nearest_labels = self._get_neighbors(x)
            proba = nearest_labels.value_counts(normalize=True)
            proba_predictions.append(proba)
        return proba_predictions

# Entraîner le modèle
def train_knn_model(df):
    X = df.drop(columns='Disease')
    y = df['Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_jaccard = JaccardKNN(n_neighbors=3)
    knn_jaccard.fit(X_train.values, y_train)
    
    y_pred_jaccard = knn_jaccard.predict(X_test.values)
    accuracy_jaccard = np.mean(y_pred_jaccard == y_test)
    print(f'Accuracy with Jaccard distance: {accuracy_jaccard}')
    print("Classification Report:")
    print(classification_report(y_test, y_pred_jaccard))
    
    return knn_jaccard, X.columns

# Encodage des symptômes
def encode_symptoms(symptoms_text, symptom_columns):
    symptoms_list = symptoms_text.split(', ')
    encoded_symptoms = pd.Series(0, index=symptom_columns)
    for symptom in symptoms_list:
        if symptom in encoded_symptoms.index:
            encoded_symptoms[symptom] = 1
    return encoded_symptoms

# Prédire les maladies
def predict_disease(knn_model, symptom_columns, new_symptoms):
    encoded_new_symptoms = encode_symptoms(new_symptoms, symptom_columns)
    new_symptoms_encoded = encoded_new_symptoms.values.reshape(1, -1)
    proba_predictions = knn_model.predict_proba(new_symptoms_encoded)
    
    diseases_prediction = proba_predictions[0].nlargest(3)
    return diseases_prediction.index.tolist()

def initialize_model(db):
    diseases_ref = db.collection('DiseasesAndSymptoms')
    docs = diseases_ref.stream()
    
    records = []
    for doc in docs:
        records.append(doc.to_dict())
    
    df = pd.DataFrame(records)
    knn_model, symptom_columns = train_knn_model(df)
    
    return knn_model, symptom_columns
