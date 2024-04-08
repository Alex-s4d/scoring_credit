import pytest
import pandas as pd
import os
import sys

# Ajoutez le chemin du répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessings import (
    application_train_test,
    bureau_and_balance,
    previous_applications,
    pos_cash,
    installments_payments,
    credit_card_balance,
    calcul_interet
)

@pytest.fixture
def sample_data():
    # Créer des données d'exemple
    sample_df = pd.DataFrame({
        'SK_ID_CURR': [1001, 1002, 1003],
        'AMT_CREDIT': [200000, 300000, 400000],
        'AMT_ANNUITY': [10000, 15000, 20000]
    })
    return sample_df



def test_application_train_test(sample_data):
    # Tester la fonction application_train_test
    df_train, df_test = application_train_test(num_rows=1000)
    assert not df_train.empty
    assert not df_test.empty
    assert 'SK_ID_CURR' in df_train.columns
    assert 'SK_ID_CURR' in df_test.columns

def test_bureau_and_balance(sample_data):
    # Tester la fonction bureau_and_balance
    bureau_df = bureau_and_balance(num_rows=1000)
    bureau_df.reset_index(inplace=True)
    assert not bureau_df.empty
    assert 'SK_ID_CURR' in bureau_df.columns

def test_previous_applications(sample_data):
    # Tester la fonction previous_applications
    prev_df = previous_applications(num_rows=1000)
    prev_df.reset_index(inplace=True)
    assert not prev_df.empty
    assert 'SK_ID_CURR' in prev_df.columns

def test_pos_cash(sample_data):
    # Tester la fonction pos_cash
    pos_df = pos_cash(num_rows=1000)
    pos_df.reset_index(inplace=True)
    assert not pos_df.empty
    assert 'SK_ID_CURR' in pos_df.columns

def test_installments_payments(sample_data):
    # Tester la fonction installments_payments
    ins_df = installments_payments(num_rows=1000)
    ins_df.reset_index(inplace=True)
    assert not ins_df.empty
    assert 'SK_ID_CURR' in ins_df.columns

def test_credit_card_balance(sample_data):
    # Tester la fonction credit_card_balance
    cc_df = credit_card_balance(num_rows=1000)
    cc_df.reset_index(inplace=True)
    assert not cc_df.empty
    assert 'SK_ID_CURR' in cc_df.columns

def test_calcul_interet(sample_data):
    # Tester la fonction calcul_interet
    sample_data_with_interest = calcul_interet(sample_data)
    assert not sample_data_with_interest.empty
    assert 'INTERET_CUMULE' in sample_data_with_interest.columns
