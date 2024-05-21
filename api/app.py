import joblib
import pandas as pd
from fastapi import FastAPI
from src.ScoringCredits import ScoringCredit
from sklearn.pipeline import Pipeline
import uvicorn

app = FastAPI()

# Charger le modèle
try:
    classifier = joblib.load('models/pipeline_final.pkl')
except Exception as e:
    print("Erreur lors du chargement du modèle :", e)

# Fonction pour obtenir les caractéristiques importantes
def get_feature_importances():
    try:
        return classifier.feature_importances_
    except AttributeError:
        return None

@app.get('/')
def index():
    return {'message': 'Welcome to the scoring credit API'}

@app.post('/predict')
def predict_scoringcredit(data: ScoringCredit):
    # Convertir les données reçues en DataFrame pandas
    input_data = pd.DataFrame([data.dict()])

    # Utiliser la pipeline pour effectuer des prédictions de probabilité
    if hasattr(classifier, 'predict_proba'):
        # Prédiction des probabilités
        probabilities = classifier.predict_proba(input_data)
        # Sélectionner la probabilité de la classe positive (classe 1)
        positive_probability = probabilities[0][1]

    # Prédiction de la classe
    prediction = classifier.predict(input_data)

    
    feature_names = input_data.columns

    feature_values_df = pd.DataFrame({
        'Feature': feature_names,
        'Values': input_data.iloc[0].values,
        })
    
    feature_values_df['Values'] = feature_values_df['Values'].round(4)
    feature_values_df['Values'] = feature_values_df['Values'].fillna(0)
    
    feature_values_dict = feature_values_df.to_dict(orient='records')

    print(feature_values_dict)

    feature_importances = None
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_

        # Suppose input_data is your DataFrame
        feature_names = input_data.columns

        # Créer un DataFrame avec les noms des features et leurs importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })

        # Trier par importance décroissante
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        feature_importance_dict = feature_importance_df.to_dict(orient='records')



    # Interprétation de la prédiction
    prediction_label = f"Risque de faillite de {round(positive_probability*100,4)}% crédit refusé" if prediction == 1 else f"Risque de faillite de {round(positive_probability*100,4)}% crédit accepté"
    #prediction_label = round(positive_probability*100)


    response = {
        "prediction": prediction_label,
        "probability": positive_probability if hasattr(classifier, 'predict_proba') else None,
        "feature_importances": feature_importance_dict,
        "feature_values": feature_values_dict,
    }

    return response


# Lancement de l'API FastAPI
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
