import pytest
import pandas as pd
import numpy as np
from src.pipeline import pipeline 
from sklearn.preprocessing import MinMaxScaler


@pytest.fixture
def sample_data():
    # Créer un échantillon de données pour les tests
    df = pd.read_csv('data/sample_test.csv')

    best_features = pd.read_csv('data/best_fetaures.csv')
    feats = best_features['feature'].unique()
    feats = np.append(feats,'INTERET_CUMULE' )
    df = df[df['TARGET'].notnull()]
    # Sélectionner les colonnes de type int et float
    numeric_columns = df[feats].select_dtypes(include=['int', 'float']).columns
    # Initialiser le MinMaxScaler
    scaler = MinMaxScaler()
    # Normaliser les colonnes sélectionnées
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    median_values = df[feats].median()
    # Imputer les valeurs manquantes avec la médiane
    df.fillna(median_values, inplace=True) 

    return df

def test_pipeline_fit_transform(sample_data):
    # Vérifier si la pipeline peut être ajustée et transformée correctement
    X = sample_data.drop(columns=['TARGET'])
    y = sample_data['TARGET']
    pipeline.fit(X, y)
    
    # Utiliser le nom correct de l'étape de transformation
    transformed_data = pipeline.named_steps['select_numeric'].transform(X)
    
    # Vérifier si la forme des données transformées est correcte
    assert transformed_data.shape[0] == len(sample_data)


def test_pipeline_predict(sample_data):
    # Vérifier si la pipeline peut prédire correctement
    new_data = sample_data.drop(columns=['TARGET'])
    predictions = pipeline.predict(new_data)
    
    # Vérifier si le nombre de prédictions est correct
    assert len(predictions) == len(new_data)
    
    # Vérifier si toutes les prédictions sont dans les limites attendues (0 ou 1)
    assert all(prediction in [0, 1] for prediction in predictions)

def test_pipeline_predict_proba(sample_data):
    # Vérifier si la pipeline peut prédire les probabilités correctement
    new_data = sample_data.drop(columns=['TARGET'])
    proba_predictions = pipeline.predict_proba(new_data)
    
    # Vérifier si le nombre de prédictions de probabilité est correct
    assert len(proba_predictions) == len(new_data)
    
    # Vérifier si toutes les prédictions de probabilité sont comprises entre 0 et 1
    for proba in proba_predictions:
        assert np.all((0 <= proba) & (proba <= 1))
