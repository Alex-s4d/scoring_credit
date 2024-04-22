import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
# Importation du package
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids




def preprocessing(Sampling=None):

    if Sampling == None:

        df = pd.read_csv('data/preprocessing_train.csv')

        best_features = pd.read_csv('data/best_fetaures.csv')
        feats = best_features['feature'].unique()
        feats = np.append(feats,'INTERET_CUMULE' )

        df = df[df['TARGET'].notnull()]
        
        X = df[feats]
        
        y = df['TARGET']
        
        # Diviser l'ensemble de données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # Sélectionner les colonnes de type int et float dans l'ensemble d'entraînement
        numeric_columns_train = X_train.select_dtypes(include=['int', 'float']).columns
        
        # Initialiser le MinMaxScaler
        scaler = MinMaxScaler()
        
        # Normaliser les colonnes sélectionnées sur l'ensemble d'entraînement
        X_train[numeric_columns_train] = scaler.fit_transform(X_train[numeric_columns_train])
        
        # Appliquer la même transformation de normalisation à l'ensemble de test
        X_test[numeric_columns_train] = scaler.transform(X_test[numeric_columns_train])
        
        # Imputer les valeurs manquantes avec la médiane de l'ensemble d'entraînement
        median_values_train = X_train.median()
        X_train.fillna(median_values_train, inplace=True)
        X_test.fillna(median_values_train, inplace=True) 

        print(X_train.shape)
        print(X_test.shape) 
        

        return X_train, X_test, y_train,y_test
    
    elif Sampling == 'Small':

        df = pd.read_csv('data/preprocessing_train.csv',nrows=300)

        best_features = pd.read_csv('data/best_fetaures.csv')
        feats = best_features['feature'].unique()
        feats = np.append(feats,'INTERET_CUMULE' )

        X = df[feats]
        
        y = df['TARGET']
        
        # Diviser l'ensemble de données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # Sélectionner les colonnes de type int et float dans l'ensemble d'entraînement
        numeric_columns_train = X_train.select_dtypes(include=['int', 'float']).columns
        
        # Initialiser le MinMaxScaler
        scaler = MinMaxScaler()
        
        # Normaliser les colonnes sélectionnées sur l'ensemble d'entraînement
        X_train[numeric_columns_train] = scaler.fit_transform(X_train[numeric_columns_train])
        
        # Appliquer la même transformation de normalisation à l'ensemble de test
        X_test[numeric_columns_train] = scaler.transform(X_test[numeric_columns_train])
        
        # Imputer les valeurs manquantes avec la médiane de l'ensemble d'entraînement
        median_values_train = X_train.median()
        X_train.fillna(median_values_train, inplace=True)
        X_test.fillna(median_values_train, inplace=True) 

        print(X_train.shape)
        print(X_test.shape) 

        return X_train, X_test, y_train, y_test
        
    
    else:
        print('parameter not recognized')

    return X_train, X_test, y_train, y_test

def create_weight():
    # Custom metric

    df = pd.read_csv('data/preprocessing_train.csv')

    best_features = pd.read_csv('data/best_fetaures.csv')
    feats = best_features['feature'].unique()
    feats = np.append(feats,'INTERET_CUMULE' )

    df = df[df['TARGET'].notnull()]


    gain_client_0 = df.groupby('TARGET')['INTERET_CUMULE'].mean()[0]
    print(f"Un client n'ayant pas fait faillite fait gagné en moyenne : {gain_client_0}")


    perte_client_1 = df.groupby('TARGET')['AMT_CREDIT'].mean()[1]
    print(f"Un client ayant fait faillite fait perdre en moyenne : {perte_client_1}")


    print(f"Un client n'ayant pas remboursé son crédit fait perdre en moyenne {round(( perte_client_1/gain_client_0),4)} plus que les clients ayant remboursé")

    # Minimisation des Faux négatifs pour maximiser les gains
    weight =  gain_client_0 / perte_client_1 


    return weight




