import pytest
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

# Ajoutez le chemin du répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import preprocessing, create_weight



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


