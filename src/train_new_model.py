import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer, f1_score, roc_auc_score, auc, roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
from collections import Counter 
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import mlflow.sklearn
import joblib
from pycaret.classification import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score, recall_score








from src.data_loader import preprocessing, create_weight

# Custom metric
def custom_fbeta_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=weight)


#Hyperoptim
def hyperoptim(X_train, y_train, model='LGBM'):

    if model=='LGBM':

        mlflow.lightgbm.autolog()

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
        scorer = make_scorer(custom_fbeta_score)
        
        # Définir l'espace des hyperparamètres à optimiser
        param_space = {
            'learning_rate': Real(0.01, 0.05, prior='log-uniform'),
            'num_leaves': Integer(10, 100),
            'subsample': Real(0.8, 1.0),
            'colsample_bytree': Real(0.8, 1.0),
            'max_depth': Integer(5, 50),
            'reg_alpha': Real(0.01, 0.1, prior='log-uniform'),
            'reg_lambda': Real(0.01, 0.1, prior='log-uniform'),
            'min_split_gain': Real(0.01, 0.05, prior='log-uniform'),
            'min_child_weight': Integer(10, 50)
        }
        
        # Initialiser le classifieur LGBM
        clf = lgb.LGBMClassifier(silent=-1, n_estimators=10000)
        
        # Initialiser BayesSearchCV avec le classifieur, l'espace des hyperparamètres et la validation croisée
        bayes_search = BayesSearchCV(
            clf,
            param_space,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=1,  # Niveau de verbosité pour l'optimisation
        )
        
        # Effectuer l'optimisation des hyperparamètres sur les données d'entraînement
        bayes_search.fit(X_train, y_train)
        
        # Obtenir les meilleurs hyperparamètres
        best_params = bayes_search.best_params_
        print("Meilleurs hyperparamètres trouvés :", best_params)
        
        # Utiliser les meilleurs hyperparamètres pour initialiser le modèle final
        best_clf = lgb.LGBMClassifier(silent=-1, **best_params)

        mlflow.lightgbm.autolog()

        return best_clf, best_params
    
    elif model == 'LogisticRegression':

        mlflow.sklearn.autolog()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_clf = LogisticRegression()

        best_params = None

        return best_clf, best_params
    
    elif model == 'DummyClassifier':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            best_clf = DummyClassifier()

            best_params = None

            return best_clf, best_params
    
    elif model == 'Pycaret':

            best_clf = joblib.load('models/my_best_pipeline.pkl')

            best_params = best_clf.get_params()

            return best_clf, best_params
    
    elif model == 'Optim':

            best_clf = joblib.load('models/lgbm_optim_model.pkl')

            best_params = best_clf.get_params()

            return best_clf, best_params
    
    elif model == 'ExtraTreesClassifier':
        mlflow.sklearn.autolog()
        best_clf = ExtraTreesClassifier()
        best_params = None
        return best_clf, best_params
    
    elif model == 'QuadraticDiscriminantAnalysis':
        mlflow.sklearn.autolog()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_clf = QuadraticDiscriminantAnalysis()
        best_params = None
        return best_clf, best_params


    

    


# Training
def imbalanced_training(X_train,X_test,y_train,y_test,best_clf,best_params, imbalance=None):

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    if imbalance==None:


        # CV :
        scores_custom = []
        scores_auc = []
        scores_f1 = []
        scores_precision = []
        scores_recall = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_fold, y_train_fold)

            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            predictions2 = best_clf.predict_proba(X_test_fold)

            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            fpr, tpr, thresholds = roc_curve(y_test_fold, predictions2[:, 1])
            score_auc = auc(fpr, tpr)
            score_precision = precision_score(y_test_fold, predictions)
            score_recall = recall_score(y_test_fold, predictions)

            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)
            scores_precision.append(score_precision)
            scores_recall.append(score_recall)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            mlflow.log_metric("Precision CV", score_precision)
            mlflow.log_metric("Recall CV", score_recall)

            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)
            print("Score Precision du fold :", score_precision)
            print("Score Recall du fold :", score_recall)

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
        print("-------------------------------------------------------")
        print("Scores Precision de validation croisée :", scores_precision)
        print("Moyenne des scores Precision :", np.mean(scores_precision))
        print("-------------------------------------------------------")
        print("Scores Recall de validation croisée :", scores_recall)
        print("Moyenne des scores Recall :", np.mean(scores_recall))

        # Test
        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)
        predictions2 = best_clf.predict_proba(X_test)

        # Calculer les scores
        beta_score = custom_fbeta_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, predictions2[:, 1])
        score_auc = auc(fpr, tpr)
        score_precision = precision_score(y_test, predictions)
        score_recall = recall_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)
        mlflow.log_metric("Precision Test", score_precision)
        mlflow.log_metric("Recall Test", score_recall)

        # Afficher les scores sur l'ensemble de test
        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")
        print("Scores Precision sur l'ensemble de test :", score_precision)
        print("-------------------------------------------------------")
        print("Scores Recall sur l'ensemble de test :", score_recall)
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
        scores_precision = []
        scores_recall = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_fold, y_train_fold)

            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            predictions2 = best_clf.predict_proba(X_test_fold)

            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            fpr, tpr, thresholds = roc_curve(y_test_fold, predictions2[:, 1])
            score_auc = auc(fpr, tpr)
            score_precision = precision_score(y_test_fold, predictions)
            score_recall = recall_score(y_test_fold, predictions)

            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)
            scores_precision.append(score_precision)
            scores_recall.append(score_recall)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            mlflow.log_metric("Precision CV", score_precision)
            mlflow.log_metric("Recall CV", score_recall)

            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)
            print("Score Precision du fold :", score_precision)
            print("Score Recall du fold :", score_recall)

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
        print("-------------------------------------------------------")
        print("Scores Precision de validation croisée :", scores_precision)
        print("Moyenne des scores Precision :", np.mean(scores_precision))
        print("-------------------------------------------------------")
        print("Scores Recall de validation croisée :", scores_recall)
        print("Moyenne des scores Recall :", np.mean(scores_recall))

        # Test
        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)
        predictions2 = best_clf.predict_proba(X_test)

        # Calculer les scores
        beta_score = custom_fbeta_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, predictions2[:, 1])
        score_auc = auc(fpr, tpr)
        score_precision = precision_score(y_test, predictions)
        score_recall = recall_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)
        mlflow.log_metric("Precision Test", score_precision)
        mlflow.log_metric("Recall Test", score_recall)

        # Afficher les scores sur l'ensemble de test
        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")
        print("Scores Precision sur l'ensemble de test :", score_precision)
        print("-------------------------------------------------------")
        print("Scores Recall sur l'ensemble de test :", score_recall)
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
        scores_precision = []
        scores_recall = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_fold, y_train_fold)

            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            predictions2 = best_clf.predict_proba(X_test_fold)

            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            fpr, tpr, thresholds = roc_curve(y_test_fold, predictions2[:, 1])
            score_auc = auc(fpr, tpr)
            score_precision = precision_score(y_test_fold, predictions)
            score_recall = recall_score(y_test_fold, predictions)

            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)
            scores_precision.append(score_precision)
            scores_recall.append(score_recall)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            mlflow.log_metric("Precision CV", score_precision)
            mlflow.log_metric("Recall CV", score_recall)

            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)
            print("Score Precision du fold :", score_precision)
            print("Score Recall du fold :", score_recall)

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
        print("-------------------------------------------------------")
        print("Scores Precision de validation croisée :", scores_precision)
        print("Moyenne des scores Precision :", np.mean(scores_precision))
        print("-------------------------------------------------------")
        print("Scores Recall de validation croisée :", scores_recall)
        print("Moyenne des scores Recall :", np.mean(scores_recall))

        # Test
        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)
        predictions2 = best_clf.predict_proba(X_test)

        # Calculer les scores
        beta_score = custom_fbeta_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, predictions2[:, 1])
        score_auc = auc(fpr, tpr)
        score_precision = precision_score(y_test, predictions)
        score_recall = recall_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)
        mlflow.log_metric("Precision Test", score_precision)
        mlflow.log_metric("Recall Test", score_recall)

        # Afficher les scores sur l'ensemble de test
        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")
        print("Scores Precision sur l'ensemble de test :", score_precision)
        print("-------------------------------------------------------")
        print("Scores Recall sur l'ensemble de test :", score_recall)
        print("-------------------------------------------------------")

        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions)
        print("Matrice de confusion:")
        print(conf_matrix)

        return best_model

    if imbalance == 'Class weight':
        def get_class_weights(y):
            counter = Counter(y)
            majority = max(counter.values())
            return {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}

        class_weights = get_class_weights(y_train)
        print(class_weights)

        best_clf = lgb.LGBMClassifier()
        best_clf.set_params(**best_params)
        best_clf = lgb.LGBMClassifier(class_weight=class_weights, is_unbalance=True)

        # CV
        scores_custom = []
        scores_auc = []
        scores_f1 = []
        scores_precision = []
        scores_recall = []

        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Adapter le modèle sur les données d'entraînement de ce fold
            best_clf.fit(X_train_fold, y_train_fold)

            # Faire des prédictions sur les données de test de ce fold
            predictions = best_clf.predict(X_test_fold)
            predictions2 = best_clf.predict_proba(X_test_fold)

            # Calculer les scores
            score_custom = custom_fbeta_score(y_test_fold, predictions)
            score_f1 = f1_score(y_test_fold, predictions)
            fpr, tpr, thresholds = roc_curve(y_test_fold, predictions2[:, 1])
            score_auc = auc(fpr, tpr)
            score_precision = precision_score(y_test_fold, predictions)
            score_recall = recall_score(y_test_fold, predictions)

            # Ajouter les scores à la liste
            scores_custom.append(score_custom)
            scores_auc.append(score_auc)
            scores_f1.append(score_f1)
            scores_precision.append(score_precision)
            scores_recall.append(score_recall)

            # Log des scores de validation croisée dans MLflow
            mlflow.log_metric("Custom Score CV", score_custom)
            mlflow.log_metric("AUC Score CV", score_auc)
            mlflow.log_metric("F1 Score CV", score_f1)
            mlflow.log_metric("Precision CV", score_precision)
            mlflow.log_metric("Recall CV", score_recall)

            # Afficher les scores pour ce fold
            print("Score custom du fold :", score_custom)
            print("Score AUC du fold :", score_auc)
            print("Score F1 du fold :", score_f1)
            print("Score Precision du fold :", score_precision)
            print("Score Recall du fold :", score_recall)

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
        print("-------------------------------------------------------")
        print("Scores Precision de validation croisée :", scores_precision)
        print("Moyenne des scores Precision :", np.mean(scores_precision))
        print("-------------------------------------------------------")
        print("Scores Recall de validation croisée :", scores_recall)
        print("Moyenne des scores Recall :", np.mean(scores_recall))

        # Test
        # Adapter le modèle sur les données d'entraînement resamplées
        best_clf.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = best_clf.predict(X_test)
        predictions2 = best_clf.predict_proba(X_test)

        # Calculer les scores
        beta_score = custom_fbeta_score(y_test, predictions)
        score_f1 = f1_score(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, predictions2[:, 1])
        score_auc = auc(fpr, tpr)
        score_precision = precision_score(y_test, predictions)
        score_recall = recall_score(y_test, predictions)

        # Log des scores sur l'ensemble de test dans MLflow
        mlflow.log_metric("Custom Score Test", beta_score)
        mlflow.log_metric("AUC Score Test", score_auc)
        mlflow.log_metric("F1 Score Test", score_f1)
        mlflow.log_metric("Precision Test", score_precision)
        mlflow.log_metric("Recall Test", score_recall)

        # Afficher les scores sur l'ensemble de test
        print(f"Score Beta sur l'ensemble de test: {beta_score}")
        print("-------------------------------------------------------")
        print("Scores AUC sur l'ensemble de test :", score_auc)
        print("-------------------------------------------------------")
        print("Scores F1 sur l'ensemble de test :", score_f1)
        print("-------------------------------------------------------")
        print("Scores Precision sur l'ensemble de test :", score_precision)
        print("-------------------------------------------------------")
        print("Scores Recall sur l'ensemble de test :", score_recall)
        print("-------------------------------------------------------")

        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y_test, predictions)
        print("Matrice de confusion:")
        print(conf_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de confusion')
        plt.xlabel('Classe prédite')
        plt.ylabel('Classe réelle')

        # Enregistrer l'image
        conf_matrix_path = 'confusion_matrix.png'
        plt.savefig(conf_matrix_path)
        plt.close()

        # Loguer l'image de la matrice de confusion dans MLflow
        mlflow.log_artifact(conf_matrix_path)

        importances = best_clf.feature_importances_
        features = X_train.columns    
        # Créer un DataFrame pour les importances des caractéristiques
        feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})    
        # Trier les importances des caractéristiques et sélectionner les 10 premières
        top_feature_importances = feature_importances.sort_values(by='Importance', ascending=False).head(10)    
        # Tracer les 10 premières importances des caractéristiques
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=top_feature_importances)
        plt.title('Top 10 des importances des caractéristiques')
        plt.xlabel('Importance')
        plt.ylabel('Caractéristique')    
        # Enregistrer l'image des importances des caractéristiques
        top_feature_importances_path = 'top_feature_importances.png'
        plt.savefig(top_feature_importances_path)
        plt.close()    
        # Loguer l'image des importances des caractéristiques dans MLflow
        mlflow.log_artifact(top_feature_importances_path)    
        return best_model




#  MLflow

experiment_name = "LGBM Hyperoptimisation"
mlflow.set_experiment(experiment_name)


with mlflow.start_run(run_name="LGBM hyperoptim"):

    weight = create_weight()
    print(weight)

    X_train, X_test, y_train, y_test = preprocessing(Sampling=None)


    best_model, best_params = hyperoptim(X_train,y_train,model='Optim')

    model = imbalanced_training(X_train,X_test,y_train,y_test,best_model,best_params, imbalance="Class weight")

    mlflow.lightgbm.log_model(model, "LGBM_model")
