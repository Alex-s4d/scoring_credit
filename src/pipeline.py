from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].values

# Charger les meilleures caractéristiques à partir du fichier CSV
best_features = pd.read_csv('data/best_fetaures.csv')
feats = best_features['feature'].unique()
feats = np.append(feats, 'INTERET_CUMULE')

# Charger le modèle à partir du fichier pickle
model = joblib.load('models/model.pkl')

# Créer la pipeline
pipeline = Pipeline([
    # Étape 1: Sélectionner les colonnes de type int et float
    ('select_numeric', DataFrameSelector(columns=feats)),
    
    # Étape 2: Normaliser les données
    ('scaler', MinMaxScaler()),
    
    # Étape 3: Imputer les valeurs manquantes avec la médiane
    ('imputer', SimpleImputer(strategy='median')),
    
    # Étape 4: Utiliser le modèle
    ('model', model)
])



def main():
    data = pd.read_csv('data/preprocessing_train.csv')
    X = data[feats]
    y = data['TARGET']

    new_data = pd.read_csv('data/preprocessing_test.csv')


    pipeline.fit(X, y)

    # Utiliser la pipeline pour effectuer des prédictions
    predictions = pipeline.predict(new_data)

    sub_preds = pipeline.predict_proba(new_data)[:, 1]

    submission_df = pd.DataFrame({'SK_ID_CURR': new_data['SK_ID_CURR'], 'TARGET': sub_preds})
    submission_df.to_csv('output/submission.csv',index=False)

    new_data['TARGET'] = predictions

    new_data['test'] = True
    data['test'] = False

    new_df = pd.concat([data, new_data])

    new_df.to_csv('output/new_df.csv',index=False)



if __name__ == "__main__":
        main()

