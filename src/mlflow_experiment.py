import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import joblib
import numpy as np

class MLFlowExperiment:
    def __init__(self, model, X_train, y_train, X_hide_test, y_hide_test, custom_fbeta_score):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_hide_test = X_hide_test
        self.y_hide_test = y_hide_test
        self.custom_fbeta_score = custom_fbeta_score

    def run_experiment(self):
        # Créer l'expérience MLflow
        experiment_name = "class_imbalance"
        mlflow.set_experiment(experiment_name)

        # Commencer l'exécution en spécifiant l'expérience MLflow
        with mlflow.start_run():
            # Créer et entraîner le modèle
            mlflow.lightgbm.autolog()
            self.model.fit(self.X_train, self.y_train)

            # Calculer les scores sur l'ensemble de validation croisée
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring=make_scorer(self.custom_fbeta_score))

            probas_pred = self.model.predict_proba(self.X_train)[:, 1]

            # Initialiser les listes pour stocker les résultats
            threshold_array = np.linspace(0, 1, 100)
            fbeta_list = []

            for threshold in threshold_array:
                # Labels prédits pour un seuil donné
                label_pred_threshold = (probas_pred > threshold).astype(int)
                # Calcul du f1 pour un seuil donné
                fbeta_threshold = self.custom_fbeta_score(y_true=self.y_train, y_pred=label_pred_threshold)
                fbeta_list.append(fbeta_threshold)

            fbeta_values = fbeta_list
            best_threshold_index = np.argmax(fbeta_values)
            best_threshold = threshold_array[best_threshold_index]

            # Calculer les prédictions sur l'ensemble de test caché
            test_predictions = (self.model.predict_proba(self.X_hide_test)[:, 1] > best_threshold).astype(int)

            # Calculer les métriques sur l'ensemble de test caché
            auc = roc_auc_score(self.y_hide_test, test_predictions)
            f1 = f1_score(self.y_hide_test, test_predictions)
            beta_score = self.custom_fbeta_score(self.y_hide_test, test_predictions)

            # Afficher les scores de validation croisée
            print("Scores de validation croisée :", cv_scores)
            print("Moyenne des scores de validation croisée :", np.mean(cv_scores))

            # Enregistrer les scores de validation croisée dans MLflow
            mlflow.log_metric("CV Mean Beta score", np.mean(cv_scores))
            for i, score in enumerate(cv_scores):
                mlflow.log_metric(f"CV Fold {i+1} Beta score", score)

            # Enregistrer les performances sur l'ensemble de test caché dans MLflow
            mlflow.log_metric("Test AUC", auc)
            mlflow.log_metric("Test F1", f1)
            mlflow.log_metric("Test Beta Score Custom", beta_score)

            # Enregistrer le modèle dans MLflow
            mlflow.lightgbm.log_model(self.model, "lightgbm_model_imbalanced")

            model_save_path = "models/lightgbm_model_imbalanced.pkl"
            joblib.dump(self.model, model_save_path)

