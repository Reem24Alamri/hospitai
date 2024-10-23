import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Getting raw data
def get_data():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(root_dir, "..", "raw_data", )

    df = pd.read_csv(os.path.join(csv_path, 'HDHI-Admission-data.csv'))

    return df

# Data Cleaning
def get_data_cleaned():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(root_dir, "..", "data", "raw_data")

    df = pd.read_csv(os.path.join(csv_path, 'HDHI-Admission-data.csv'))

    # drop columns pt1
    df.drop(columns=[
        'SNO',
        'MRD No.',
        'D.O.A',
        'D.O.D',
        'BNP',
        'duration of intensive unit stay',
        'month year'
    ], inplace=True)
    df = df.replace('EMPTY',np.nan)

    # imputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df[['EF', 'GLUCOSE', 'TLC', 'PLATELETS', 'HB', 'CREATININE', 'UREA']])
    df[['EF', 'GLUCOSE', 'TLC', 'PLATELETS', 'HB', 'CREATININE', 'UREA']] = imputer.transform(df[['EF', 'GLUCOSE', 'TLC', 'PLATELETS', 'HB', 'CREATININE', 'UREA']])

    # filter the expiry outcome and drop it
    df = df[df['OUTCOME'] == 'DISCHARGE']
    df.drop(columns='OUTCOME', inplace=True)

    # cleaning 'CHEST INFECTION'
    df = df[df['CHEST INFECTION'].isin(['1', '0'])]
    df['CHEST INFECTION'] = df['CHEST INFECTION'].astype('int')

    X = df.drop(columns='DURATION OF STAY')
    y = df['DURATION OF STAY']

    return (X, y)
