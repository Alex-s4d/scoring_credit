import requests
from src.ScoringCredits import ScoringCredit 
import json
import pandas as pd

def main(new_data):
    #new_data = pd.read_csv('data/preprocessing_test.csv')
    first_row = new_data.iloc[[0]]
    # Créer une instance de la classe ScoringCredit avec les données de la première ligne
    scoring_credit_instance = ScoringCredit(
        ACTIVE_AMT_ANNUITY_MEAN=first_row['ACTIVE_AMT_ANNUITY_MEAN'].values[0],
        ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN=first_row['ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN'].values[0],
        ACTIVE_AMT_CREDIT_SUM_DEBT_SUM=first_row['ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'].values[0],
        ACTIVE_AMT_CREDIT_SUM_MEAN=first_row['ACTIVE_AMT_CREDIT_SUM_MEAN'].values[0],
        ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN=first_row['ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN'].values[0],
        ACTIVE_CREDIT_DAY_OVERDUE_MEAN=first_row['ACTIVE_CREDIT_DAY_OVERDUE_MEAN'].values[0],
        ACTIVE_DAYS_CREDIT_ENDDATE_MIN=first_row['ACTIVE_DAYS_CREDIT_ENDDATE_MIN'].values[0],
        ACTIVE_DAYS_CREDIT_MEAN=first_row['ACTIVE_DAYS_CREDIT_MEAN'].values[0],
        ACTIVE_DAYS_CREDIT_UPDATE_MEAN=first_row['ACTIVE_DAYS_CREDIT_UPDATE_MEAN'].values[0],
        ACTIVE_DAYS_CREDIT_VAR=first_row['ACTIVE_DAYS_CREDIT_VAR'].values[0],
        ACTIVE_MONTHS_BALANCE_SIZE_MEAN=first_row['ACTIVE_MONTHS_BALANCE_SIZE_MEAN'].values[0],
        AMT_ANNUITY=first_row['AMT_ANNUITY'].values[0],
        ANNUITY_INCOME_PERC=first_row['ANNUITY_INCOME_PERC'].values[0],
        APPROVED_AMT_ANNUITY_MAX=first_row['APPROVED_AMT_ANNUITY_MAX'].values[0],
        APPROVED_AMT_GOODS_PRICE_MIN=first_row['APPROVED_AMT_GOODS_PRICE_MIN'].values[0],
        APPROVED_APP_CREDIT_PERC_MEAN=first_row['APPROVED_APP_CREDIT_PERC_MEAN'].values[0],
        APPROVED_APP_CREDIT_PERC_VAR=first_row['APPROVED_APP_CREDIT_PERC_VAR'].values[0],
        APPROVED_CNT_PAYMENT_MEAN=first_row['APPROVED_CNT_PAYMENT_MEAN'].values[0],
        BASEMENTAREA_AVG=first_row['BASEMENTAREA_AVG'].values[0],
        BURO_AMT_CREDIT_MAX_OVERDUE_MEAN=first_row['BURO_AMT_CREDIT_MAX_OVERDUE_MEAN'].values[0],
        BURO_AMT_CREDIT_SUM_LIMIT_SUM=first_row['BURO_AMT_CREDIT_SUM_LIMIT_SUM'].values[0],
        BURO_AMT_CREDIT_SUM_MEAN=first_row['BURO_AMT_CREDIT_SUM_MEAN'].values[0],
        BURO_AMT_CREDIT_SUM_SUM=first_row['BURO_AMT_CREDIT_SUM_SUM'].values[0],
        BURO_CREDIT_ACTIVE_Active_MEAN=first_row['BURO_CREDIT_ACTIVE_Active_MEAN'].values[0],
        BURO_CREDIT_ACTIVE_Closed_MEAN=first_row['BURO_CREDIT_ACTIVE_Closed_MEAN'].values[0],
        BURO_CREDIT_ACTIVE_Sold_MEAN=first_row['BURO_CREDIT_ACTIVE_Sold_MEAN'].values[0],
        BURO_CREDIT_TYPE_Credit_card_MEAN=first_row['BURO_CREDIT_TYPE_Credit_card_MEAN'].values[0],
        BURO_CREDIT_TYPE_Microloan_MEAN=first_row['BURO_CREDIT_TYPE_Microloan_MEAN'].values[0],
        BURO_CREDIT_TYPE_Mortgage_MEAN=first_row['BURO_CREDIT_TYPE_Mortgage_MEAN'].values[0],
        BURO_DAYS_CREDIT_ENDDATE_MAX=first_row['BURO_DAYS_CREDIT_ENDDATE_MAX'].values[0],
        BURO_DAYS_CREDIT_ENDDATE_MIN=first_row['BURO_DAYS_CREDIT_ENDDATE_MIN'].values[0],
        BURO_DAYS_CREDIT_MAX=first_row['BURO_DAYS_CREDIT_MAX'].values[0],
        BURO_DAYS_CREDIT_UPDATE_MEAN=first_row['BURO_DAYS_CREDIT_UPDATE_MEAN'].values[0],
        BURO_DAYS_CREDIT_VAR=first_row['BURO_DAYS_CREDIT_VAR'].values[0],
        BURO_MONTHS_BALANCE_SIZE_SUM=first_row['BURO_MONTHS_BALANCE_SIZE_SUM'].values[0],
        BURO_STATUS_0_MEAN_MEAN=first_row['BURO_STATUS_0_MEAN_MEAN'].values[0],
        BURO_STATUS_1_MEAN_MEAN=first_row['BURO_STATUS_1_MEAN_MEAN'].values[0],
        BURO_STATUS_C_MEAN_MEAN=first_row['BURO_STATUS_C_MEAN_MEAN'].values[0],
        BURO_STATUS_X_MEAN_MEAN=first_row['BURO_STATUS_X_MEAN_MEAN'].values[0],
        CC_AMT_BALANCE_MEAN=first_row['CC_AMT_BALANCE_MEAN'].values[0],
        CC_AMT_BALANCE_MIN=first_row['CC_AMT_BALANCE_MIN'].values[0],
        CC_AMT_DRAWINGS_ATM_CURRENT_MAX=first_row['CC_AMT_DRAWINGS_ATM_CURRENT_MAX'].values[0],
        CC_AMT_DRAWINGS_CURRENT_SUM=first_row['CC_AMT_DRAWINGS_CURRENT_SUM'].values[0],
        CC_AMT_DRAWINGS_OTHER_CURRENT_MIN=first_row['CC_AMT_DRAWINGS_OTHER_CURRENT_MIN'].values[0],
        CC_AMT_DRAWINGS_POS_CURRENT_VAR=first_row['CC_AMT_DRAWINGS_POS_CURRENT_VAR'].values[0],
        CC_AMT_RECIVABLE_MIN=first_row['CC_AMT_RECIVABLE_MIN'].values[0],
        CC_AMT_TOTAL_RECEIVABLE_MEAN=first_row['CC_AMT_TOTAL_RECEIVABLE_MEAN'].values[0],
        CC_CNT_DRAWINGS_ATM_CURRENT_SUM=first_row['CC_CNT_DRAWINGS_ATM_CURRENT_SUM'].values[0],
        CC_CNT_DRAWINGS_CURRENT_MAX=first_row['CC_CNT_DRAWINGS_CURRENT_MAX'].values[0],
        CC_CNT_DRAWINGS_CURRENT_MEAN=first_row['CC_CNT_DRAWINGS_CURRENT_MEAN'].values[0],
        CC_CNT_DRAWINGS_OTHER_CURRENT_VAR=first_row['CC_CNT_DRAWINGS_OTHER_CURRENT_VAR'].values[0],
        CC_NAME_CONTRACT_STATUS_Demand_MIN=first_row['CC_NAME_CONTRACT_STATUS_Demand_MIN'].values[0],
        CC_NAME_CONTRACT_STATUS_Sent_proposal_MAX=first_row['CC_NAME_CONTRACT_STATUS_Sent_proposal_MAX'].values[0],
        CC_SK_DPD_DEF_MAX=first_row['CC_SK_DPD_DEF_MAX'].values[0],
        CC_SK_DPD_SUM=first_row['CC_SK_DPD_SUM'].values[0],
        CLOSED_AMT_ANNUITY_MEAN=first_row['CLOSED_AMT_ANNUITY_MEAN'].values[0],
        CLOSED_AMT_CREDIT_SUM_DEBT_MAX=first_row['CLOSED_AMT_CREDIT_SUM_DEBT_MAX'].values[0],
        CLOSED_AMT_CREDIT_SUM_LIMIT_SUM=first_row['CLOSED_AMT_CREDIT_SUM_LIMIT_SUM'].values[0],
        CLOSED_AMT_CREDIT_SUM_MEAN=first_row['CLOSED_AMT_CREDIT_SUM_MEAN'].values[0],
        CLOSED_CREDIT_DAY_OVERDUE_MEAN=first_row['CLOSED_CREDIT_DAY_OVERDUE_MEAN'].values[0],
        CLOSED_DAYS_CREDIT_ENDDATE_MAX=first_row['CLOSED_DAYS_CREDIT_ENDDATE_MAX'].values[0],
        CLOSED_DAYS_CREDIT_ENDDATE_MIN=first_row['CLOSED_DAYS_CREDIT_ENDDATE_MIN'].values[0],
        CLOSED_DAYS_CREDIT_MAX=first_row['CLOSED_DAYS_CREDIT_MAX'].values[0],
        CLOSED_DAYS_CREDIT_VAR=first_row['CLOSED_DAYS_CREDIT_VAR'].values[0],
        CODE_GENDER=first_row['CODE_GENDER'].values[0],
        COMMONAREA_AVG=first_row['COMMONAREA_AVG'].values[0],
        DAYS_BIRTH=first_row['DAYS_BIRTH'].values[0],
        DAYS_EMPLOYED=first_row['DAYS_EMPLOYED'].values[0],
        DAYS_ID_PUBLISH=first_row['DAYS_ID_PUBLISH'].values[0],
        DAYS_LAST_PHONE_CHANGE=first_row['DAYS_LAST_PHONE_CHANGE'].values[0],
        DAYS_REGISTRATION=first_row['DAYS_REGISTRATION'].values[0],
        DEF_30_CNT_SOCIAL_CIRCLE=first_row['DEF_30_CNT_SOCIAL_CIRCLE'].values[0],
        ELEVATORS_AVG=first_row['ELEVATORS_AVG'].values[0],
        EXT_SOURCE_1=first_row['EXT_SOURCE_1'].values[0],
        EXT_SOURCE_2=first_row['EXT_SOURCE_2'].values[0],
        EXT_SOURCE_3=first_row['EXT_SOURCE_3'].values[0],
        FLAG_DOCUMENT_3=first_row['FLAG_DOCUMENT_3'].values[0],
        FLOORSMAX_AVG=first_row['FLOORSMAX_AVG'].values[0],
        FLOORSMIN_MEDI=first_row['FLOORSMIN_MEDI'].values[0],
        FONDKAPREMONT_MODE_reg_oper_account=first_row['FONDKAPREMONT_MODE_reg_oper_account'].values[0],
        INCOME_CREDIT_PERC=first_row['INCOME_CREDIT_PERC'].values[0],
        INCOME_PER_PERSON=first_row['INCOME_PER_PERSON'].values[0],
        INSTAL_AMT_INSTALMENT_MEAN=first_row['INSTAL_AMT_INSTALMENT_MEAN'].values[0],
        INSTAL_AMT_PAYMENT_MAX=first_row['INSTAL_AMT_PAYMENT_MAX'].values[0],
        INSTAL_COUNT=first_row['INSTAL_COUNT'].values[0],
        INSTAL_DBD_MAX=first_row['INSTAL_DBD_MAX'].values[0],
        INSTAL_DBD_MEAN=first_row['INSTAL_DBD_MEAN'].values[0],
        INSTAL_DPD_MEAN=first_row['INSTAL_DPD_MEAN'].values[0],
        INSTAL_DPD_SUM=first_row['INSTAL_DPD_SUM'].values[0],
        INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE=first_row['INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE'].values[0],
        INSTAL_PAYMENT_DIFF_MAX=first_row['INSTAL_PAYMENT_DIFF_MAX'].values[0],
        INSTAL_PAYMENT_DIFF_MEAN=first_row['INSTAL_PAYMENT_DIFF_MEAN'].values[0],
        INSTAL_PAYMENT_PERC_SUM=first_row['INSTAL_PAYMENT_PERC_SUM'].values[0],
        LANDAREA_AVG=first_row['LANDAREA_AVG'].values[0],
        LIVE_REGION_NOT_WORK_REGION=first_row['LIVE_REGION_NOT_WORK_REGION'].values[0],
        LIVINGAPARTMENTS_AVG=first_row['LIVINGAPARTMENTS_AVG'].values[0],
        NAME_CONTRACT_TYPE_Cash_loans=first_row['NAME_CONTRACT_TYPE_Cash_loans'].values[0],
        NAME_EDUCATION_TYPE_Higher_education=first_row['NAME_EDUCATION_TYPE_Higher_education'].values[0],
        NAME_FAMILY_STATUS_Married=first_row['NAME_FAMILY_STATUS_Married'].values[0],
        NAME_INCOME_TYPE_Working=first_row['NAME_INCOME_TYPE_Working'].values[0],
        NONLIVINGAPARTMENTS_MODE=first_row['NONLIVINGAPARTMENTS_MODE'].values[0],
        NONLIVINGAREA_AVG=first_row['NONLIVINGAREA_AVG'].values[0],
        OBS_30_CNT_SOCIAL_CIRCLE=first_row['OBS_30_CNT_SOCIAL_CIRCLE'].values[0],
        OWN_CAR_AGE=first_row['OWN_CAR_AGE'].values[0],
        PAYMENT_RATE=first_row['PAYMENT_RATE'].values[0],
        POS_COUNT=first_row['POS_COUNT'].values[0],
        POS_MONTHS_BALANCE_MAX=first_row['POS_MONTHS_BALANCE_MAX'].values[0],
        POS_MONTHS_BALANCE_SIZE=first_row['POS_MONTHS_BALANCE_SIZE'].values[0],
        POS_NAME_CONTRACT_STATUS_Active_MEAN=first_row['POS_NAME_CONTRACT_STATUS_Active_MEAN'].values[0],
        POS_NAME_CONTRACT_STATUS_Completed_MEAN=first_row['POS_NAME_CONTRACT_STATUS_Completed_MEAN'].values[0],
        POS_NAME_CONTRACT_STATUS_Signed_MEAN=first_row['POS_NAME_CONTRACT_STATUS_Signed_MEAN'].values[0],
        POS_SK_DPD_DEF_MEAN=first_row['POS_SK_DPD_DEF_MEAN'].values[0],
        PREV_AMT_APPLICATION_MIN=first_row['PREV_AMT_APPLICATION_MIN'].values[0],
        PREV_AMT_DOWN_PAYMENT_MEAN=first_row['PREV_AMT_DOWN_PAYMENT_MEAN'].values[0],
        PREV_AMT_GOODS_PRICE_MEAN=first_row['PREV_AMT_GOODS_PRICE_MEAN'].values[0],
        PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN=first_row['PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN'].values[0],
        PREV_CHANNEL_TYPE_Contact_center_MEAN=first_row['PREV_CHANNEL_TYPE_Contact_center_MEAN'].values[0],
        PREV_CHANNEL_TYPE_Country_wide_MEAN=first_row['PREV_CHANNEL_TYPE_Country_wide_MEAN'].values[0],
        PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN=first_row['PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN'].values[0],
        PREV_CNT_PAYMENT_SUM=first_row['PREV_CNT_PAYMENT_SUM'].values[0],
        PREV_CODE_REJECT_REASON_HC_MEAN=first_row['PREV_CODE_REJECT_REASON_HC_MEAN'].values[0],
        PREV_CODE_REJECT_REASON_SCOFR_MEAN=first_row['PREV_CODE_REJECT_REASON_SCOFR_MEAN'].values[0],
        PREV_CODE_REJECT_REASON_XAP_MEAN=first_row['PREV_CODE_REJECT_REASON_XAP_MEAN'].values[0],
        PREV_DAYS_DECISION_MAX=first_row['PREV_DAYS_DECISION_MAX'].values[0],
        PREV_DAYS_DECISION_MIN=first_row['PREV_DAYS_DECISION_MIN'].values[0],
        PREV_HOUR_APPR_PROCESS_START_MEAN=first_row['PREV_HOUR_APPR_PROCESS_START_MEAN'].values[0],
        PREV_NAME_CLIENT_TYPE_New_MEAN=first_row['PREV_NAME_CLIENT_TYPE_New_MEAN'].values[0],
        PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN=first_row['PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN'].values[0],
        PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN=first_row['PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN'].values[0],
        PREV_NAME_GOODS_CATEGORY_Clothing_and_Accessories_MEAN=first_row['PREV_NAME_GOODS_CATEGORY_Clothing_and_Accessories_MEAN'].values[0],
        PREV_NAME_PAYMENT_TYPE_XNA_MEAN=first_row['PREV_NAME_PAYMENT_TYPE_XNA_MEAN'].values[0],
        PREV_NAME_PRODUCT_TYPE_walk_in_MEAN=first_row['PREV_NAME_PRODUCT_TYPE_walk_in_MEAN'].values[0],
        PREV_NAME_PRODUCT_TYPE_x_sell_MEAN=first_row['PREV_NAME_PRODUCT_TYPE_x_sell_MEAN'].values[0],
        PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN=first_row['PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN'].values[0],
        PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN=first_row['PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN'].values[0],
        PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN=first_row['PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN'].values[0],
        PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN=first_row['PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN'].values[0],
        PREV_NAME_TYPE_SUITE_nan_MEAN=first_row['PREV_NAME_TYPE_SUITE_nan_MEAN'].values[0],
        PREV_NAME_YIELD_GROUP_XNA_MEAN=first_row['PREV_NAME_YIELD_GROUP_XNA_MEAN'].values[0],
        PREV_NAME_YIELD_GROUP_high_MEAN=first_row['PREV_NAME_YIELD_GROUP_high_MEAN'].values[0],
        PREV_NAME_YIELD_GROUP_low_normal_MEAN=first_row['PREV_NAME_YIELD_GROUP_low_normal_MEAN'].values[0],
        PREV_NAME_YIELD_GROUP_middle_MEAN=first_row['PREV_NAME_YIELD_GROUP_middle_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Card_Street_MEAN=first_row['PREV_PRODUCT_COMBINATION_Card_Street_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Card_X_Sell_MEAN=first_row['PREV_PRODUCT_COMBINATION_Card_X_Sell_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Cash_MEAN=first_row['PREV_PRODUCT_COMBINATION_Cash_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Cash_Street__high_MEAN=first_row['PREV_PRODUCT_COMBINATION_Cash_Street__high_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN=first_row['PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN=first_row['PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN=first_row['PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN=first_row['PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN'].values[0],
        PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN=first_row['PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN'].values[0],
        PREV_RATE_DOWN_PAYMENT_MEAN=first_row['PREV_RATE_DOWN_PAYMENT_MEAN'].values[0],
        REFUSED_AMT_CREDIT_MAX=first_row['REFUSED_AMT_CREDIT_MAX'].values[0],
        REFUSED_AMT_DOWN_PAYMENT_MAX=first_row['REFUSED_AMT_DOWN_PAYMENT_MAX'].values[0],
        REFUSED_APP_CREDIT_PERC_MEAN=first_row['REFUSED_APP_CREDIT_PERC_MEAN'].values[0],
        REFUSED_APP_CREDIT_PERC_VAR=first_row['REFUSED_APP_CREDIT_PERC_VAR'].values[0],
        REFUSED_DAYS_DECISION_MAX=first_row['REFUSED_DAYS_DECISION_MAX'].values[0],
        REFUSED_RATE_DOWN_PAYMENT_MIN=first_row['REFUSED_RATE_DOWN_PAYMENT_MIN'].values[0],
        REGION_RATING_CLIENT_W_CITY=first_row['REGION_RATING_CLIENT_W_CITY'].values[0],
        REG_CITY_NOT_WORK_CITY=first_row['REG_CITY_NOT_WORK_CITY'].values[0],
        INTERET_CUMULE=first_row['INTERET_CUMULE'].values[0]
        )
    scoring_credit_dict = scoring_credit_instance.dict()



    for key, value in scoring_credit_dict.items():
        if isinstance(value, float):
            # Limiter la précision des valeurs flottantes à 6 chiffres après la virgule
            scoring_credit_dict[key] = "{:.6f}".format(value)

    # Définissez l'URL de votre API
    url = 'http://13.38.9.787:8000/predict'

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
