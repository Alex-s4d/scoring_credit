import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

        # Sélectionner les colonnes de type int et float
        numeric_columns = df[feats].select_dtypes(include=['int', 'float']).columns

        # Initialiser le MinMaxScaler
        scaler = MinMaxScaler()

        # Normaliser les colonnes sélectionnées
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        median_values = df[feats].median()

        # Imputer les valeurs manquantes avec la médiane
        df.fillna(median_values, inplace=True) 

        X = df[feats]

        y = df['TARGET']

        # Diviser l'ensemble de données en ensembles d'entraînement, de validation et de test
        X_train, X_hide_test, y_train, y_hide_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Diviser X_temp et y_temp pour obtenir le X_validation, y_validation, X_test, y_test
        #X_hide_test, X_test, y_hide_test, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        print(X_train.shape)
        #print(X_test.shape)
        print(X_hide_test.shape)

        return X_train, X_hide_test, y_train, y_hide_test

    elif Sampling == 'SMOTE':

        df = pd.read_csv('data/preprocessing_train.csv')

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

        X = df[feats]

        y = df['TARGET']

        # Diviser l'ensemble de données en ensembles d'entraînement, de validation et de test
        X_train, X_hide_test, y_train, y_hide_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Diviser X_temp et y_temp pour obtenir le X_validation, y_validation, X_test, y_test
        #X_hide_test, X_test, y_hide_test, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        print(X_train.shape)
        #print(X_test.shape)
        print(X_hide_test.shape)

        counter = Counter(y_train)
        print(f'Répartition de la target avant SMOTE : {counter}')
              
        smote = SMOTE(random_state=2)

        # Appliquer SMOTE sur les données
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        counter = Counter(y_resampled)
        print(f'Répartition de la target Après SMOTE : {counter}')


        return X_resampled , X_hide_test, y_resampled, y_hide_test
    

    elif Sampling == 'Undersampling':

        df = pd.read_csv('data/preprocessing_train.csv')

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

        X = df[feats]

        y = df['TARGET']

        # Diviser l'ensemble de données en ensembles d'entraînement, de validation et de test
        X_train, X_hide_test, y_train, y_hide_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        print(X_train.shape)
        #print(X_test.shape)
        print(X_hide_test.shape)

        counter = Counter(y_train)
        print(f'Répartition de la target avant Undersampling: {counter}')
              
        smote = SMOTE(random_state=42)

        # Appliquer Undersampling sur les données
        cc = ClusterCentroids(sampling_strategy='auto', random_state=42)


        X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

        counter = Counter(y_resampled)
        print(f'Répartition de la target Après Undersampling : {counter}')


        return X_resampled, X_hide_test, y_resampled, y_hide_test 
    
    if Sampling == 'Small':

        df = pd.read_csv('data/preprocessing_train.csv')

        best_features = pd.read_csv('data/best_fetaures.csv')
        feats = best_features['feature'].unique()
        feats = np.append(feats,'INTERET_CUMULE' )

        df = df[df['TARGET'].notnull()][0:300]

        # Sélectionner les colonnes de type int et float
        numeric_columns = df[feats].select_dtypes(include=['int', 'float']).columns

        # Initialiser le MinMaxScaler
        scaler = MinMaxScaler()

        # Normaliser les colonnes sélectionnées
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        median_values = df[feats].median()

        # Imputer les valeurs manquantes avec la médiane
        df.fillna(median_values, inplace=True) 

        X = df[feats]

        y = df['TARGET']

        # Diviser l'ensemble de données en ensembles d'entraînement, de validation et de test
        X_train, X_hide_test, y_train, y_hide_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Diviser X_temp et y_temp pour obtenir le X_validation, y_validation, X_test, y_test
        #X_hide_test, X_test, y_hide_test, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        print(X_train.shape)
        #print(X_test.shape)
        print(X_hide_test.shape)

        return X_train, X_hide_test, y_train, y_hide_test
        
    
    else:
        print('parameter not recognized')

    return X_train, X_hide_test, y_train, y_hide_test

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


#X_train, X_test, X_hide_test, y_train, y_test, y_hide_test = preprocessing()




