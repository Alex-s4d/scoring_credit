import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.simplefilter(action='ignore', category=FutureWarning)

df_train = pd.read_csv('data/preprocessing_train.csv')
df_test = pd.read_csv('data/preprocessing_test.csv')

def clean_feature_names(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df

def main(df_train,df_test, num_folds, stratified=False, debug=False):
    train_df = df_train.copy()
    test_df = df_test.copy()
    train_df = clean_feature_names(train_df)
    test_df = clean_feature_names(test_df)
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    gc.collect()

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Définir le nom de l'expérience
    experiment_name = "Baseline P7 OCR"

    # Spécifier l'expérience MLflow
    mlflow.set_experiment(experiment_name)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # Commencer l'exécution
        with mlflow.start_run():
            # Enregistrer les paramètres
            mlflow.log_param("nthread", 4)
            mlflow.log_param("n_estimators", 10000)
            mlflow.log_param("learning_rate", 0.02)
            mlflow.log_param("num_leaves", 34)
            mlflow.log_param("colsample_bytree", 0.9497036)
            mlflow.log_param("subsample", 0.8715623)
            mlflow.log_param("max_depth", 8)
            mlflow.log_param("reg_alpha", 0.041545473)
            mlflow.log_param("reg_lambda", 0.0735294)
            mlflow.log_param("min_split_gain", 0.0222415)
            mlflow.log_param("min_child_weight", 39.3259775)
            mlflow.log_param("silent", -1)

            clf = LGBMClassifier(
                nthread=4,
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=34,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,)

            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric='auc')

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

            # Enregistrer l'AUC dans MLflow
            mlflow.log_metric("AUC", roc_auc_score(valid_y, oof_preds[valid_idx]))

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv('output/submission_kernel02.csv', index=False)
    display_importances(feature_importance_df)

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    best_features.to_csv("data/best_fetaures.csv",index=False)
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


if __name__ == "__main__":
    feat_importance = main(df_train,df_test, num_folds= 5, stratified= False, debug= 'debug')
