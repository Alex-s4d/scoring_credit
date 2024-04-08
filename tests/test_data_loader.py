import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

# Ajoutez le chemin du répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_weight():
    # Custom metric

    df = pd.read_csv('data/sample_test.csv')

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



# Test pour s'assurer que la fonction create_weight retourne une valeur numérique
def test_create_weight_output_type():
    weight = create_weight()
    assert isinstance(weight, (int, float))

# Test pour vérifier si le poids calculé est positif
def test_create_weight_positive():
    weight = create_weight()
    assert weight > 0

# Test pour vérifier si le poids calculé est raisonnable
def test_create_weight_reasonable():
    weight = create_weight()
    assert weight < 1000


