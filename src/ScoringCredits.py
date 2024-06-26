from pydantic import BaseModel, Field
from typing import Optional

class ScoringCredit(BaseModel):
    ACTIVE_AMT_ANNUITY_MEAN: Optional[float]
    ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN: Optional[float]
    ACTIVE_AMT_CREDIT_SUM_DEBT_SUM: Optional[float]
    ACTIVE_AMT_CREDIT_SUM_MEAN: Optional[float]
    ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN: Optional[float]
    ACTIVE_CREDIT_DAY_OVERDUE_MEAN: Optional[float]
    ACTIVE_DAYS_CREDIT_ENDDATE_MIN: Optional[float]
    ACTIVE_DAYS_CREDIT_MEAN: Optional[float]
    ACTIVE_DAYS_CREDIT_UPDATE_MEAN: Optional[float]
    ACTIVE_DAYS_CREDIT_VAR: Optional[float]
    ACTIVE_MONTHS_BALANCE_SIZE_MEAN: Optional[float]
    AMT_ANNUITY: Optional[float]
    ANNUITY_INCOME_PERC: Optional[float]
    APPROVED_AMT_ANNUITY_MAX: Optional[float]
    APPROVED_AMT_GOODS_PRICE_MIN: Optional[float]
    APPROVED_APP_CREDIT_PERC_MEAN: Optional[float]
    APPROVED_APP_CREDIT_PERC_VAR: Optional[float]
    APPROVED_CNT_PAYMENT_MEAN: Optional[float]
    BASEMENTAREA_AVG: Optional[float]
    BURO_AMT_CREDIT_MAX_OVERDUE_MEAN: Optional[float]
    BURO_AMT_CREDIT_SUM_LIMIT_SUM: Optional[float]
    BURO_AMT_CREDIT_SUM_MEAN: Optional[float]
    BURO_AMT_CREDIT_SUM_SUM: Optional[float]
    BURO_CREDIT_ACTIVE_Active_MEAN: Optional[float]
    BURO_CREDIT_ACTIVE_Closed_MEAN: Optional[float]
    BURO_CREDIT_ACTIVE_Sold_MEAN: Optional[float]
    BURO_CREDIT_TYPE_Credit_card_MEAN: Optional[float]
    BURO_CREDIT_TYPE_Microloan_MEAN: Optional[float]
    BURO_CREDIT_TYPE_Mortgage_MEAN: Optional[float]
    BURO_DAYS_CREDIT_ENDDATE_MAX: Optional[float]
    BURO_DAYS_CREDIT_ENDDATE_MIN: Optional[float]
    BURO_DAYS_CREDIT_MAX: Optional[float]
    BURO_DAYS_CREDIT_UPDATE_MEAN: Optional[float]
    BURO_DAYS_CREDIT_VAR: Optional[float]
    BURO_MONTHS_BALANCE_SIZE_SUM: Optional[float]
    BURO_STATUS_0_MEAN_MEAN: Optional[float]
    BURO_STATUS_1_MEAN_MEAN: Optional[float]
    BURO_STATUS_C_MEAN_MEAN: Optional[float]
    BURO_STATUS_X_MEAN_MEAN: Optional[float]
    CC_AMT_BALANCE_MEAN: Optional[float]
    CC_AMT_BALANCE_MIN: Optional[float]
    CC_AMT_DRAWINGS_ATM_CURRENT_MAX: Optional[float]
    CC_AMT_DRAWINGS_CURRENT_SUM: Optional[float]
    CC_AMT_DRAWINGS_OTHER_CURRENT_MIN: Optional[float]
    CC_AMT_DRAWINGS_POS_CURRENT_VAR: Optional[float]
    CC_AMT_RECIVABLE_MIN: Optional[float]
    CC_AMT_TOTAL_RECEIVABLE_MEAN: Optional[float]
    CC_CNT_DRAWINGS_ATM_CURRENT_SUM: Optional[float]
    CC_CNT_DRAWINGS_CURRENT_MAX: Optional[float]
    CC_CNT_DRAWINGS_CURRENT_MEAN: Optional[float]
    CC_CNT_DRAWINGS_OTHER_CURRENT_VAR: Optional[float]
    CC_NAME_CONTRACT_STATUS_Demand_MIN: Optional[float]
    CC_NAME_CONTRACT_STATUS_Sent_proposal_MAX: Optional[float]
    CC_SK_DPD_DEF_MAX: Optional[float]
    CC_SK_DPD_SUM: Optional[float]
    CLOSED_AMT_ANNUITY_MEAN: Optional[float]
    CLOSED_AMT_CREDIT_SUM_DEBT_MAX: Optional[float]
    CLOSED_AMT_CREDIT_SUM_LIMIT_SUM: Optional[float]
    CLOSED_AMT_CREDIT_SUM_MEAN: Optional[float]
    CLOSED_CREDIT_DAY_OVERDUE_MEAN: Optional[float]
    CLOSED_DAYS_CREDIT_ENDDATE_MAX: Optional[float]
    CLOSED_DAYS_CREDIT_ENDDATE_MIN: Optional[float]
    CLOSED_DAYS_CREDIT_MAX: Optional[float]
    CLOSED_DAYS_CREDIT_VAR: Optional[float]
    CODE_GENDER: Optional[float]
    COMMONAREA_AVG: Optional[float]
    DAYS_BIRTH: Optional[int]
    DAYS_EMPLOYED: Optional[float]
    DAYS_ID_PUBLISH: Optional[int]
    DAYS_LAST_PHONE_CHANGE: Optional[float]
    DAYS_REGISTRATION: Optional[float]
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float]
    ELEVATORS_AVG: Optional[float]
    EXT_SOURCE_1: Optional[float]
    EXT_SOURCE_2: Optional[float]
    EXT_SOURCE_3: Optional[float]
    FLAG_DOCUMENT_3: Optional[int]
    FLOORSMAX_AVG: Optional[float]
    FLOORSMIN_MEDI: Optional[float]
    FONDKAPREMONT_MODE_reg_oper_account: Optional[float]
    INCOME_CREDIT_PERC: Optional[float]
    INCOME_PER_PERSON: Optional[float]
    INSTAL_AMT_INSTALMENT_MEAN: Optional[float]
    INSTAL_AMT_PAYMENT_MAX: Optional[float]
    INSTAL_COUNT: Optional[float]
    INSTAL_DBD_MAX: Optional[float]
    INSTAL_DBD_MEAN: Optional[float]
    INSTAL_DPD_MEAN: Optional[float]
    INSTAL_DPD_SUM: Optional[float]
    INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE: Optional[float]
    INSTAL_PAYMENT_DIFF_MAX: Optional[float]
    INSTAL_PAYMENT_DIFF_MEAN: Optional[float]
    INSTAL_PAYMENT_PERC_SUM: Optional[float]
    LANDAREA_AVG: Optional[float]
    LIVE_REGION_NOT_WORK_REGION: Optional[float]
    LIVINGAPARTMENTS_AVG: Optional[float]
    NAME_CONTRACT_TYPE_Cash_loans: Optional[float]
    NAME_EDUCATION_TYPE_Higher_education: Optional[float]
    NAME_FAMILY_STATUS_Married: Optional[float]
    NAME_INCOME_TYPE_Working: Optional[float]
    NONLIVINGAPARTMENTS_MODE: Optional[float]
    NONLIVINGAREA_AVG: Optional[float]
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float]
    OWN_CAR_AGE: Optional[float]
    PAYMENT_RATE: Optional[float]
    POS_COUNT: Optional[float]
    POS_MONTHS_BALANCE_MAX: Optional[float]
    POS_MONTHS_BALANCE_SIZE: Optional[float]
    POS_NAME_CONTRACT_STATUS_Active_MEAN: Optional[float]
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: Optional[float]
    POS_NAME_CONTRACT_STATUS_Signed_MEAN: Optional[float]
    POS_SK_DPD_DEF_MEAN: Optional[float]
    PREV_AMT_APPLICATION_MIN: Optional[float]
    PREV_AMT_DOWN_PAYMENT_MEAN: Optional[float]
    PREV_AMT_GOODS_PRICE_MEAN: Optional[float]
    PREV_CHANNEL_TYPE_AP___Cash_loan__MEAN: Optional[float]
    PREV_CHANNEL_TYPE_Contact_center_MEAN: Optional[float]
    PREV_CHANNEL_TYPE_Country_wide_MEAN: Optional[float]
    PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN: Optional[float]
    PREV_CNT_PAYMENT_SUM: Optional[float]
    PREV_CODE_REJECT_REASON_HC_MEAN: Optional[float]
    PREV_CODE_REJECT_REASON_SCOFR_MEAN: Optional[float]
    PREV_CODE_REJECT_REASON_XAP_MEAN: Optional[float]
    PREV_DAYS_DECISION_MAX: Optional[float]
    PREV_DAYS_DECISION_MIN: Optional[float]
    PREV_HOUR_APPR_PROCESS_START_MEAN: Optional[float]
    PREV_NAME_CLIENT_TYPE_New_MEAN: Optional[float]
    PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN: Optional[float]
    PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN: Optional[float]
    PREV_NAME_GOODS_CATEGORY_Clothing_and_Accessories_MEAN: Optional[float]
    PREV_NAME_PAYMENT_TYPE_XNA_MEAN: Optional[float]
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN: Optional[float]
    PREV_NAME_PRODUCT_TYPE_x_sell_MEAN: Optional[float]
    PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN: Optional[float]
    PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN: Optional[float]
    PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN: Optional[float]
    PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN: Optional[float]
    PREV_NAME_TYPE_SUITE_nan_MEAN: Optional[float]
    PREV_NAME_YIELD_GROUP_XNA_MEAN: Optional[float]
    PREV_NAME_YIELD_GROUP_high_MEAN: Optional[float]
    PREV_NAME_YIELD_GROUP_low_normal_MEAN: Optional[float]
    PREV_NAME_YIELD_GROUP_middle_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Card_Street_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Card_X_Sell_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Cash_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Cash_Street__high_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Cash_Street__middle_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Cash_X_Sell__high_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_Cash_X_Sell__low_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN: Optional[float]
    PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN: Optional[float]
    PREV_RATE_DOWN_PAYMENT_MEAN: Optional[float]
    REFUSED_AMT_CREDIT_MAX: Optional[float]
    REFUSED_AMT_DOWN_PAYMENT_MAX: Optional[float]
    REFUSED_APP_CREDIT_PERC_MEAN: Optional[float]
    REFUSED_APP_CREDIT_PERC_VAR: Optional[float]
    REFUSED_DAYS_DECISION_MAX: Optional[float]
    REFUSED_RATE_DOWN_PAYMENT_MIN: Optional[float]
    REGION_RATING_CLIENT_W_CITY: Optional[float]
    REG_CITY_NOT_WORK_CITY: Optional[float]
    INTERET_CUMULE: Optional[float]