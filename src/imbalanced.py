import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer, f1_score, roc_auc_score
from skopt import BayesSearchCV
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from data_loader import preprocessing



from data_loader import preprocessing, create_weight


weight = create_weight()

X_train, X_hide_test, y_train, y_hide_test = preprocessing(Sampling='Small')

# Custom metric


def custom_fbeta_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=weight)




# Créer l'expérience MLflow
experiment_name = "class imbalance"
mlflow.set_experiment(experiment_name)


model = lgb.LGBMClassifier(
    nthread=4,
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=34,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.041545473,
    reg_lambda=0.0735294,
    min_split_gain=0.0222415,
    min_child_weight=39.3259775,
    silent=-1,
    class_weight='balanced',
    is_unbalance= True)


def grid_search(X_train, y_train):

    # Créer le classificateur
    clf = LGBMClassifier(nthread=4, silent=-1)

    # Définir l'espace de recherche des hyperparamètres
    param_grid = {
        'n_estimators': [100, 1000],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [20, 50],
        'colsample_bytree': [0.8, 1.0],
        'subsample': [0.8, 1.0],
        'max_depth': [5, 15],
        'reg_alpha': [0.01, 0.5],
        'reg_lambda': [0.01, 0.5],
        'min_split_gain': [0.01, 0.5],
        'min_child_weight': [1, 100],
    }

    # Créer la grille de recherche
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=make_scorer(custom_fbeta_score), n_jobs=-1)

    # Exécuter la recherche sur grille sur les données d'entraînement
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés
    print("Meilleurs paramètres :", grid_search.best_params_)

    # Afficher la meilleure performance
    print("Meilleure performance :", grid_search.best_score_)

    # Récupérer le meilleur modèle
    best_model = grid_search.best_estimator_

    return best_model


#model = grid_search(X_train,y_train)


# Créer l'expérience MLflow
experiment_name = "class imbalance"
mlflow.set_experiment(experiment_name)

# Commencer l'exécution en spécifiant l'expérience MLflow
with mlflow.start_run():
    # Enregistrer les paramètres de la recherche sur grille
    #mlflow.log_params(grid_search.best_params_)

    # Créer et entraîner le modèle avec le meilleur jeu de paramètres
    mlflow.lightgbm.autolog()
    model.fit(X_train, y_train)

    # Calculer les scores sur l'ensemble de validation croisée
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(custom_fbeta_score))

    probas_pred = model.predict_proba(X_train)[:, 1]

    # Initialiser les listes pour stocker les résultats
    threshold_array = np.linspace(0, 1, 100)
    fbeta_list = []


    for threshold in threshold_array:
        # Labels prédits pour un seuil donné
        label_pred_threshold = (probas_pred > threshold).astype(int)
        # Calcul du f1 pour un seuil donné
        fbeta_threshold = custom_fbeta_score(
            y_true=y_train, y_pred=label_pred_threshold
        )

        fbeta_list.append(fbeta_threshold)
        
    #fbeta_values = [item[1] for item in fbeta_list]
    fbeta_values = fbeta_list


    best_threshold_index = np.argmax(fbeta_values)

    # Récupérer le seuil correspondant
    best_threshold = threshold_array[best_threshold_index]

    # Calculer les prédictions sur l'ensemble de test caché
    #test_predictions = best_model.predict(X_hide_test)
    test_predictions = (model.predict_proba(X_hide_test)[:, 1] > best_threshold).astype(int)

    # Calculer les métriques sur l'ensemble de test caché
    auc = roc_auc_score(y_hide_test, test_predictions)
    f1 = f1_score(y_hide_test, test_predictions)
    beta_score = custom_fbeta_score(y_hide_test, test_predictions)

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
    mlflow.lightgbm.log_model(model, "lightgbm_model_SMALL")
