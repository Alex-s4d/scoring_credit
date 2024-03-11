from src import preprocessings
from src.preprocessings import timer
from src.data_loader import preprocessing, create_weight
from src.mlflow_experiment import MLFlowExperiment
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score
import uvicorn
from src import baseline_LGBM
from src import pipeline
from api import app, predict
import subprocess
import time
import multiprocessing


if __name__ == '__main__':
    with timer("Full model run"):
        preprocessings.main()

        # Optional : create best_features.csv :
        #baseline_LGBM.main() 


        # create weight : 
        weight = create_weight()

        # Choice Sampling : 'None', 'Small', 'SMOTE', 'Undersampling' :
        X_train, X_hide_test, y_train, y_hide_test = preprocessing(Sampling='Small')



        # Define metric custom : 
        def custom_fbeta_score(y_true, y_pred):
            return fbeta_score(y_true, y_pred, beta=weight)
        
        # Define model : 
        model = LGBMClassifier(
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


        #MLflow experiment : 
        mlflow_exp = MLFlowExperiment(model=model, 
                              X_train=X_train, y_train=y_train, 
                              X_hide_test=X_hide_test, y_hide_test=y_hide_test, 
                              custom_fbeta_score=custom_fbeta_score)
        
        mlflow_exp.run_experiment()

        pipeline.main()
    









        










    