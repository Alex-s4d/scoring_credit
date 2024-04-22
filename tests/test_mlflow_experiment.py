import pytest
import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from src.mlflow_experiment import MLFlowExperiment

@pytest.fixture
def setup_experiment():
    # Créer des données de test
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser les paramètres
    model = RandomForestClassifier(random_state=42)
    custom_fbeta_score = f1_score
    experiment = MLFlowExperiment(model, X_train, y_train, X_test, y_test, custom_fbeta_score)

    return experiment

def test_experiment_run(setup_experiment):
    experiment = setup_experiment
    experiment.run_experiment()

    # Vérifier si le modèle a été entraîné
    assert hasattr(experiment, 'model')
    assert hasattr(experiment.model, 'predict')
