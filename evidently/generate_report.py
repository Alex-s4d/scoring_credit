
import joblib
import pandas as pd
import numpy as np
import evidently
from datetime import datetime, timedelta


from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *

def create_report_drift(data):
    
    reference = data[data['test']==False][0:10000]
    current = data[data['test']==True][0:10000]

    data_report = Report(
        metrics=[
            DataDriftPreset(),
        ],
    )

    data_report.run(reference_data=reference, current_data=current)
    return data_report

def create_report_quality(data):
    
    reference = data[data['test']==False][0:10000]
    current = data[data['test']==True][0:10000]

    data_report = Report(
        metrics=[
            DataQualityPreset(),
        ],
    )

    data_report.run(reference_data=reference, current_data=current)
    return data_report

def create_tests(data):
    
    reference = data[data['test']==False][0:10000]
    current = data[data['test']==True][0:10000]
    
    suite = TestSuite(tests=[
    NoTargetPerformanceTestPreset(),
    ])

    suite.run(reference_data=reference, current_data=current)
    return suite

def generate_report(df):
    try:
        report = create_report_drift(df)
        report.save_html(f"report_drift.html")
        print("drift report done")
    except:
        print("error for drift report")
            
    try:
        report = create_report_quality(df)
        report.save_html(f"report_quality.html")
        print("quality report done")
    except:
        print("error for quality report")
        
    try :
        test = create_tests(df)
        test.save_html(f"report_test.html")
        print("test report done")
    except:
        print(f" error for test report")

data = pd.read_csv('../output/new_df.csv')

if __name__ == "__main__":
        generate_report(data)
