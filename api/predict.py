import requests
from src.ScoringCredits import ScoringCredit 
import json
import pandas as pd

def main(new_data):
    #new_data = pd.read_csv('data/preprocessing_test.csv')

    first_row = new_data.iloc[[0]]

    # Créer une instance de la classe ScoringCredit avec les données de la première ligne
    scoring_credit_instance = ScoringCredit(
        AMT_INCOME_TOTAL=first_row['AMT_INCOME_TOTAL'].values[0],
        AMT_CREDIT=first_row['AMT_CREDIT'].values[0],
        AMT_ANNUITY=first_row['AMT_ANNUITY'].values[0],
        AMT_GOODS_PRICE=first_row['AMT_GOODS_PRICE'].values[0],
        REGION_POPULATION_RELATIVE=first_row['REGION_POPULATION_RELATIVE'].values[0],
        DAYS_BIRTH=first_row['DAYS_BIRTH'].values[0],
        DAYS_EMPLOYED=first_row['DAYS_EMPLOYED'].values[0],
        DAYS_REGISTRATION=first_row['DAYS_REGISTRATION'].values[0],
        DAYS_ID_PUBLISH=first_row['DAYS_ID_PUBLISH'].values[0],
        OWN_CAR_AGE=first_row['OWN_CAR_AGE'].values[0],
        HOUR_APPR_PROCESS_START=first_row['HOUR_APPR_PROCESS_START'].values[0],
        EXT_SOURCE_1=first_row['EXT_SOURCE_1'].values[0],
        EXT_SOURCE_2=first_row['EXT_SOURCE_2'].values[0],
        EXT_SOURCE_3=first_row['EXT_SOURCE_3'].values[0],
        TOTALAREA_MODE=first_row['TOTALAREA_MODE'].values[0],
        DAYS_LAST_PHONE_CHANGE=first_row['DAYS_LAST_PHONE_CHANGE'].values[0],
        DAYS_EMPLOYED_PERC=first_row['DAYS_EMPLOYED_PERC'].values[0],
        INCOME_CREDIT_PERC=first_row['INCOME_CREDIT_PERC'].values[0],
        INCOME_PER_PERSON=first_row['INCOME_PER_PERSON'].values[0],
        ANNUITY_INCOME_PERC=first_row['ANNUITY_INCOME_PERC'].values[0],
        PAYMENT_RATE=first_row['PAYMENT_RATE'].values[0],
        BURO_DAYS_CREDIT_MIN=first_row['BURO_DAYS_CREDIT_MIN'].values[0],
        BURO_DAYS_CREDIT_MAX=first_row['BURO_DAYS_CREDIT_MAX'].values[0],
        BURO_DAYS_CREDIT_MEAN=first_row['BURO_DAYS_CREDIT_MEAN'].values[0],
        BURO_DAYS_CREDIT_VAR=first_row['BURO_DAYS_CREDIT_VAR'].values[0],
        BURO_DAYS_CREDIT_ENDDATE_MIN=first_row['BURO_DAYS_CREDIT_ENDDATE_MIN'].values[0],
        BURO_DAYS_CREDIT_ENDDATE_MAX=first_row['BURO_DAYS_CREDIT_ENDDATE_MAX'].values[0],
        BURO_DAYS_CREDIT_ENDDATE_MEAN=first_row['BURO_DAYS_CREDIT_ENDDATE_MEAN'].values[0],
        BURO_DAYS_CREDIT_UPDATE_MEAN=first_row['BURO_DAYS_CREDIT_UPDATE_MEAN'].values[0],
        BURO_AMT_CREDIT_MAX_OVERDUE_MEAN=first_row['BURO_AMT_CREDIT_MAX_OVERDUE_MEAN'].values[0],
        BURO_AMT_CREDIT_SUM_MAX=first_row['BURO_AMT_CREDIT_SUM_MAX'].values[0],
        BURO_AMT_CREDIT_SUM_MEAN=first_row['BURO_AMT_CREDIT_SUM_MEAN'].values[0],
        BURO_AMT_CREDIT_SUM_SUM=first_row['BURO_AMT_CREDIT_SUM_SUM'].values[0],
        BURO_AMT_CREDIT_SUM_DEBT_MEAN=first_row['BURO_AMT_CREDIT_SUM_DEBT_MEAN'].values[0],
        ACTIVE_DAYS_CREDIT_MIN=first_row['ACTIVE_DAYS_CREDIT_MIN'].values[0],
        ACTIVE_DAYS_CREDIT_MAX=first_row['ACTIVE_DAYS_CREDIT_MAX'].values[0],
        ACTIVE_DAYS_CREDIT_MEAN=first_row['ACTIVE_DAYS_CREDIT_MEAN'].values[0],
        ACTIVE_DAYS_CREDIT_VAR=first_row['ACTIVE_DAYS_CREDIT_VAR'].values[0],
        ACTIVE_DAYS_CREDIT_ENDDATE_MIN=first_row['ACTIVE_DAYS_CREDIT_ENDDATE_MIN'].values[0],
        ACTIVE_DAYS_CREDIT_ENDDATE_MAX=first_row['ACTIVE_DAYS_CREDIT_ENDDATE_MAX'].values[0],
        ACTIVE_DAYS_CREDIT_ENDDATE_MEAN=first_row['ACTIVE_DAYS_CREDIT_ENDDATE_MEAN'].values[0],
        ACTIVE_DAYS_CREDIT_UPDATE_MEAN=first_row['ACTIVE_DAYS_CREDIT_UPDATE_MEAN'].values[0],
        ACTIVE_AMT_CREDIT_SUM_MAX=first_row['ACTIVE_AMT_CREDIT_SUM_MAX'].values[0],
        ACTIVE_AMT_CREDIT_SUM_MEAN=first_row['ACTIVE_AMT_CREDIT_SUM_MEAN'].values[0],
        ACTIVE_AMT_CREDIT_SUM_SUM=first_row['ACTIVE_AMT_CREDIT_SUM_SUM'].values[0],
        ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN=first_row['ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN'].values[0],
        CLOSED_DAYS_CREDIT_MIN=first_row['CLOSED_DAYS_CREDIT_MIN'].values[0],
        CLOSED_DAYS_CREDIT_MAX=first_row['CLOSED_DAYS_CREDIT_MAX'].values[0],
        CLOSED_DAYS_CREDIT_MEAN=first_row['CLOSED_DAYS_CREDIT_MEAN'].values[0],
        CLOSED_DAYS_CREDIT_VAR=first_row['CLOSED_DAYS_CREDIT_VAR'].values[0],
        CLOSED_DAYS_CREDIT_ENDDATE_MIN=first_row['CLOSED_DAYS_CREDIT_ENDDATE_MIN'].values[0],
        CLOSED_DAYS_CREDIT_ENDDATE_MAX=first_row['CLOSED_DAYS_CREDIT_ENDDATE_MAX'].values[0],
        CLOSED_DAYS_CREDIT_ENDDATE_MEAN=first_row['CLOSED_DAYS_CREDIT_ENDDATE_MEAN'].values[0],
        CLOSED_DAYS_CREDIT_UPDATE_MEAN=first_row['CLOSED_DAYS_CREDIT_UPDATE_MEAN'].values[0],
        CLOSED_AMT_CREDIT_SUM_MAX=first_row['CLOSED_AMT_CREDIT_SUM_MAX'].values[0],
        CLOSED_AMT_CREDIT_SUM_MEAN=first_row['CLOSED_AMT_CREDIT_SUM_MEAN'].values[0],
        CLOSED_AMT_CREDIT_SUM_SUM=first_row['CLOSED_AMT_CREDIT_SUM_SUM'].values[0],
        PREV_AMT_ANNUITY_MIN=first_row['PREV_AMT_ANNUITY_MIN'].values[0],
        PREV_AMT_ANNUITY_MEAN=first_row['PREV_AMT_ANNUITY_MEAN'].values[0],
        PREV_AMT_CREDIT_MEAN=first_row['PREV_AMT_CREDIT_MEAN'].values[0],
        PREV_APP_CREDIT_PERC_MIN=first_row['PREV_APP_CREDIT_PERC_MIN'].values[0],
        PREV_APP_CREDIT_PERC_MEAN=first_row['PREV_APP_CREDIT_PERC_MEAN'].values[0],
        PREV_APP_CREDIT_PERC_VAR=first_row['PREV_APP_CREDIT_PERC_VAR'].values[0],
        PREV_HOUR_APPR_PROCESS_START_MEAN=first_row['PREV_HOUR_APPR_PROCESS_START_MEAN'].values[0],
        PREV_RATE_DOWN_PAYMENT_MEAN=first_row['PREV_RATE_DOWN_PAYMENT_MEAN'].values[0],
        PREV_DAYS_DECISION_MAX=first_row['PREV_DAYS_DECISION_MAX'].values[0],
        PREV_DAYS_DECISION_MEAN=first_row['PREV_DAYS_DECISION_MEAN'].values[0],
        PREV_CNT_PAYMENT_MEAN=first_row['PREV_CNT_PAYMENT_MEAN'].values[0],
        PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN=first_row['PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN'].values[0],
        PREV_NAME_TYPE_SUITE_nan_MEAN=first_row['PREV_NAME_TYPE_SUITE_nan_MEAN'].values[0],
        PREV_NAME_YIELD_GROUP_middle_MEAN=first_row['PREV_NAME_YIELD_GROUP_middle_MEAN'].values[0],
        APPROVED_AMT_ANNUITY_MIN=first_row['APPROVED_AMT_ANNUITY_MIN'].values[0],
        APPROVED_AMT_ANNUITY_MAX=first_row['APPROVED_AMT_ANNUITY_MAX'].values[0],
        APPROVED_AMT_ANNUITY_MEAN=first_row['APPROVED_AMT_ANNUITY_MEAN'].values[0],
        APPROVED_AMT_CREDIT_MIN=first_row['APPROVED_AMT_CREDIT_MIN'].values[0],
        APPROVED_APP_CREDIT_PERC_MIN=first_row['APPROVED_APP_CREDIT_PERC_MIN'].values[0],
        APPROVED_APP_CREDIT_PERC_MEAN=first_row['APPROVED_APP_CREDIT_PERC_MEAN'].values[0],
        APPROVED_APP_CREDIT_PERC_VAR=first_row['APPROVED_APP_CREDIT_PERC_VAR'].values[0],
        APPROVED_HOUR_APPR_PROCESS_START_MEAN=first_row['APPROVED_HOUR_APPR_PROCESS_START_MEAN'].values[0],
        APPROVED_DAYS_DECISION_MAX=first_row['APPROVED_DAYS_DECISION_MAX'].values[0],
        APPROVED_DAYS_DECISION_MEAN=first_row['APPROVED_DAYS_DECISION_MEAN'].values[0],
        APPROVED_CNT_PAYMENT_MEAN=first_row['APPROVED_CNT_PAYMENT_MEAN'].values[0],
        POS_MONTHS_BALANCE_MEAN=first_row['POS_MONTHS_BALANCE_MEAN'].values[0],
        POS_MONTHS_BALANCE_SIZE=first_row['POS_MONTHS_BALANCE_SIZE'].values[0],
        POS_NAME_CONTRACT_STATUS_Active_MEAN=first_row['POS_NAME_CONTRACT_STATUS_Active_MEAN'].values[0],
        POS_NAME_CONTRACT_STATUS_Completed_MEAN=first_row['POS_NAME_CONTRACT_STATUS_Completed_MEAN'].values[0],
        INSTAL_DPD_MEAN=first_row['INSTAL_DPD_MEAN'].values[0],
        INSTAL_DBD_MAX=first_row['INSTAL_DBD_MAX'].values[0],
        INSTAL_DBD_MEAN=first_row['INSTAL_DBD_MEAN'].values[0],
        INSTAL_DBD_SUM=first_row['INSTAL_DBD_SUM'].values[0],
        INSTAL_PAYMENT_PERC_VAR=first_row['INSTAL_PAYMENT_PERC_VAR'].values[0],
        INSTAL_AMT_INSTALMENT_MAX=first_row['INSTAL_AMT_INSTALMENT_MAX'].values[0],
        INSTAL_AMT_INSTALMENT_MEAN=first_row['INSTAL_AMT_INSTALMENT_MEAN'].values[0],
        INSTAL_AMT_PAYMENT_MIN=first_row['INSTAL_AMT_PAYMENT_MIN'].values[0],
        INSTAL_AMT_PAYMENT_MAX=first_row['INSTAL_AMT_PAYMENT_MAX'].values[0],
        INSTAL_AMT_PAYMENT_MEAN=first_row['INSTAL_AMT_PAYMENT_MEAN'].values[0],
        INSTAL_AMT_PAYMENT_SUM=first_row['INSTAL_AMT_PAYMENT_SUM'].values[0],
        INSTAL_DAYS_ENTRY_PAYMENT_MAX=first_row['INSTAL_DAYS_ENTRY_PAYMENT_MAX'].values[0],
        INSTAL_DAYS_ENTRY_PAYMENT_MEAN=first_row['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'].values[0],
        INSTAL_DAYS_ENTRY_PAYMENT_SUM=first_row['INSTAL_DAYS_ENTRY_PAYMENT_SUM'].values[0],
        INTERET_CUMULE=first_row['INTERET_CUMULE'].values[0]
    )

    scoring_credit_dict = scoring_credit_instance.dict()



    for key, value in scoring_credit_dict.items():
        if isinstance(value, float):
            # Limiter la précision des valeurs flottantes à 6 chiffres après la virgule
            scoring_credit_dict[key] = "{:.6f}".format(value)

    # Définissez l'URL de votre API
    url = 'http://127.0.0.1:8000/predict'

    # Envoyez la requête POST à votre API avec les données JSON
    response = requests.post(url, json=scoring_credit_dict)

    # Vérifiez si la requête a réussi
    if response.status_code == 200:
        # Affichez la réponse de l'API
        print(response.json())
    else:
        # Affichez le code d'erreur en cas d'échec de la requête
        print("Erreur lors de la requête :", response.status_code)

    return response.json()

if __name__ == "__main__":
        main()
