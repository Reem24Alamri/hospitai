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
        'duration of intensive unit stay'
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

    # creation of sin and cos features
    month_dict = {
        'Jan':1,
        'Feb':2,
        'Mar':3,
        'Apr':4,
        'May':5,
        'Jun':6,
        'Jul':7,
        'Aug':8,
        'Sep':9,
        'Oct':10,
        'Nov':11,
        'Dec':12
    }
    df['month_nb'] = df['month year'].apply(lambda x: month_dict[x[:3]])
    months_in_a_year = 12
    df['sin_admission'] = np.sin(2 * np.pi * (df['month_nb'] - 1) / months_in_a_year)
    df['cos_admission'] = np.cos(2 * np.pi * (df['month_nb'] - 1) / months_in_a_year)
    df.drop(columns=['month year', 'month_nb'], inplace=True)

    return df
