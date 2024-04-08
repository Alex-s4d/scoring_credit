import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
from collections import Counter 




from src.data_loader2 import preprocessing, create_weight

# Custom metric
def custom_fbeta_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=weight)

#Hyperoptim
def hyperoptim(X_train, y_train):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scorer = make_scorer(custom_fbeta_score)
    
    # Définir l'espace des hyperparamètres à optimiser
    param_space = {
        'n_estimators': Integer(100, 1000),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'num_leaves': Integer(20, 50),
        'subsample': Real(0.8, 1.0),
        'colsample_bytree': Real(0.8, 1.0),
        'max_depth': Integer(5, 10),
        'reg_alpha': Real(0.01, 0.1, prior='log-uniform'),
        'reg_lambda': Real(0.01, 0.1, prior='log-uniform'),
        'min_split_gain': Real(0.01, 0.1, prior='log-uniform'),
        'min_child_weight': Integer(1, 20)
    }
    
    # Initialiser le classifieur LGBM
    clf = lgb.LGBMClassifier(silent=-1)
    
    # Initialiser BayesSearchCV avec le classifieur, l'espace des hyperparamètres et la validation croisée
    bayes_search = BayesSearchCV(clf, param_space, cv=cv, scoring='f1', n_jobs=-1)
    
    # Effectuer l'optimisation des hyperparamètres sur les données d'entraînement
    bayes_search.fit(X_train, y_train)
    
    # Obtenir les meilleurs hyperparamètres
    best_params = bayes_search.best_params_
    print("Meilleurs hyperparamètres trouvés :", best_params)
    
    # Utiliser les meilleurs hyperparamètres pour initialiser le modèle final
    best_clf = lgb.LGBMClassifier(silent=-1, **best_params)

    return best_clf, best_params


# Training
def imbalanced_training(X_train,X_test,y_train,y_test,best_clf,best_params, imbalance=None):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if imbalance==None:


        # CV :
        scores_custom = []
        scores_auc = []
        scores_f1 = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_fold, y_train_fold)
            
            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            
            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_auc = roc_auc_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            
            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            
            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)

        # Afficher les scores de validation croisée
        print("-------------------------------------------------------")
        print("Scores custom de validation croisée :", scores_custom)
        print("Moyenne des scores custom :", np.mean(scores_custom))
        print("-------------------------------------------------------")
        print("Scores AUC de validation croisée :", scores_auc)
        print("Moyenne des scores AUC :", np.mean(scores_auc))
        print("-------------------------------------------------------")
        print("Scores F1 de validation croisée :", scores_f1)
        print("Moyenne des scores F1 :", np.mean(scores_f1))

        # Test
        best_clf.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)

        # Calculer le score Beta
        beta_score = custom_fbeta_score(y_test, predictions)
        score_auc = roc_auc_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)

        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")


        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions)
        print("Matrice de confusion:")
        print(conf_matrix)


        return best_model
    
    if imbalance == 'SMOTE':

        #CV
        scores_custom = []
        scores_auc = []
        scores_f1 = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            # Appliquer SMOTE sur les données d'entraînement de ce fold
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
            
            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_resampled, y_train_resampled)
            
            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            
            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_auc = roc_auc_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            
            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            
            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)

        # Afficher les scores de validation croisée
        print("-------------------------------------------------------")
        print("Scores custom de validation croisée :", scores_custom)
        print("Moyenne des scores custom :", np.mean(scores_custom))
        print("-------------------------------------------------------")
        print("Scores AUC de validation croisée :", scores_auc)
        print("Moyenne des scores AUC :", np.mean(scores_auc))
        print("-------------------------------------------------------")
        print("Scores F1 de validation croisée :", scores_f1)
        print("Moyenne des scores F1 :", np.mean(scores_f1))

        #test
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train_resampled, y_train_resampled)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)

        # Calcul des scores
        beta_score = custom_fbeta_score(y_test, predictions)
        score_auc = roc_auc_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)

        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")


        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions)
        print("Matrice de confusion:")
        print(conf_matrix)


        return best_model

    elif imbalance == 'Undersampling':

        # CV
        scores_custom = []
        scores_auc = []
        scores_f1 = []

        oss = OneSidedSelection(random_state=42)

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            # Appliquer l'undersampling sur les données d'entraînement de ce fold
            X_resampled, y_resampled = oss.fit_resample(X_train_fold, y_train_fold)
            
            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_resampled, y_train_resampled)
            
            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            
            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_auc = roc_auc_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            
            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            
            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)

        # Afficher les scores de validation croisée
        print("-------------------------------------------------------")
        print("Scores custom de validation croisée :", scores_custom)
        print("Moyenne des scores custom :", np.mean(scores_custom))
        print("-------------------------------------------------------")
        print("Scores AUC de validation croisée :", scores_auc)
        print("Moyenne des scores AUC :", np.mean(scores_auc))
        print("-------------------------------------------------------")
        print("Scores F1 de validation croisée :", scores_f1)
        print("Moyenne des scores F1 :", np.mean(scores_f1))

        # Test
        X_train_resampled, y_train_resampled = oss.fit_resample(X_test, y_train)

        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train_resampled, y_train_resampled)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)

        # Calcul des scores
        beta_score = custom_fbeta_score(y_test, predictions)
        score_auc = roc_auc_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)

        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")


        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions)
        print("Matrice de confusion:")
        print(conf_matrix)


        return best_model

    elif imbalance == 'Class weight':

        def get_class_weights(y):
            counter = Counter(y)
            majority = max(counter.values())
            return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}

        class_weights = get_class_weights(y_train)
        print(class_weights)

        best_clf = lgb.LGBMClassifier(class_weight=class_weights,is_unbalance= True, **best_params)

        #CV
        scores_custom = []
        scores_auc = []
        scores_f1 = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            
            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_fold, y_train_fold)
            
            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            
            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_auc = roc_auc_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            
            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            
            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)

        # Afficher les scores de validation croisée
        print("-------------------------------------------------------")
        print("Scores custom de validation croisée :", scores_custom)
        print("Moyenne des scores custom :", np.mean(scores_custom))
        print("-------------------------------------------------------")
        print("Scores AUC de validation croisée :", scores_auc)
        print("Moyenne des scores AUC :", np.mean(scores_auc))
        print("-------------------------------------------------------")
        print("Scores F1 de validation croisée :", scores_f1)
        print("Moyenne des scores F1 :", np.mean(scores_f1))

        #Test
        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)

        # Calculer le score Beta
        beta_score = custom_fbeta_score(y_test, predictions)
        score_auc = roc_auc_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)

        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")


        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions)
        print("Matrice de confusion:")
        print(conf_matrix)

        return best_model




#  MLflow

experiment_name = "LGBM SMOTE"
mlflow.set_experiment(experiment_name)


with mlflow.start_run():



    mlflow.lightgbm.autolog()

    weight = create_weight()

    X_train, X_test, y_train, y_test = preprocessing(Sampling=None)

    best_model, best_params = hyperoptim(X_train,y_train)

    model = imbalanced_training(X_train,X_test,y_train,y_test,best_model,best_params, imbalance='SMOTE')

    mlflow.lightgbm.log_model(model, "lightgbm_model")
