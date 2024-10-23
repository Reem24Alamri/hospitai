from fastapi import FastAPI
from pydantic import BaseModel
from hospital.ml_logic.data import get_data_cleaned
from hospital.ml_logic.preprocessing import sk_learn_proc
from hospital.ml_logic.registry import load_model
from hospital.ml_logic.model import evaluate_model
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

app = FastAPI()

app.state.model = load_model()

# Default values for the new top important features
default_values = {
    'ALCOHOL': 0,
    'CAD': 1,
    'PRIOR CMP': 0,
    'CKD': 0,
    'RAISED CARDIAC ENZYMES': 0,
    'HFREF': 0,
    'SMOKING ':0,
    'HEART FAILURE':0,
    'AKI':0,
    'CVA INFRACT':0,
    'PULMONARY EMBOLISM':0,
    'DM':0,
    'HTN':0,
    'CARDIOGENIC SHOCK':0,
    'SEVERE ANAEMIA':0,
    'PLATELETS': 150,
    'EF': 60,
    'RURAL': 'U',
    'HFNEF': 0,
    'VALVULAR': 0,
    'CHB': 0,
    'SSS': 0,
    'CVA BLEED': 0,
    'AF': 0,
    'VT': 0,
    'PSVT': 0,
    'CONGENITAL': 0,
    'UTI': 0,
    'NEURO CARDIOGENIC SYNCOPE': 0,
    'ORTHOSTATIC': 0,
    'INFECTIVE ENDOCARDITIS': 0,
    'DVT': 0,
    'SHOCK': 0,
    'CHEST INFECTION': 0,
    'STABLE ANGINA': 0,
    'RAISED CARDIAC ENZYMES': 1,

    }


def create_df(conversion_dict, age, GENDER, TYPE_OF_ADMISSION, UREA, TLC, STABLE_ANGINA, ACS, CREATININE, ATYPICAL_CHEST_PAIN, ANAEMIA, STEMI, GLUCOSE, HB):
    # Constructing the dictionary with new features
    df_dict = {'AGE': age,
                       'UREA':UREA,
        'TLC': TLC,
                'CREATININE': CREATININE,
                        'GLUCOSE':GLUCOSE,
        'HB': HB

               }
    features = [GENDER, TYPE_OF_ADMISSION,  STABLE_ANGINA, ACS, ATYPICAL_CHEST_PAIN, ANAEMIA, STEMI, ]

    for k, f in zip(conversion_dict.keys(), features):
        print(f)
        df_dict[k] = [conversion_dict[k][f]] if k in conversion_dict else [f]

    combined_dict = df_dict | default_values

    # Create the DataFrame
    return pd.DataFrame(combined_dict)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/testing")
def testing():
    model = app.state.model

    conversion_dict = {
        'GENDER': {'Male': 'M', 'Female': 'F'},
        'TYPE_OF_ADMISSION': {'EMERGENCY': 'E', 'OPD': 'O'},
        'UREA':{27},
        'TLC': {7.8},
        'STABLE_ANGINA': {'Yes' :1 , 'NO':0},
        'ACS': {'Yes' :1 , 'NO':0},
        'CREATININE': {1.2},
        'ATYPICAL_CHEST_PAIN' : {'Yes' :1 , 'NO':0},
        'ANAEMIA': {'Yes' :1 , 'NO':0},
        'STEMI' : {'Yes' :1 , 'NO':0},
        'GLUCOSE':{120},
        'HB': {13.0}
    }

    # Creating a new dataframe using updated important features
    new_df = create_df(
        conversion_dict, 30,'Male', 'EMERGENCY', 27, 7.8, 'NO',  'NO', 1.2,  'NO',  'NO',  'NO', 120, 13.0,
    )

    # Preprocessing
    proc_new = sk_learn_proc(new_df)

    # Predicting using the model
    return {"Prediction": f"{model.predict(proc_new)[0]}"}


@app.get("/predict")
def read_item(AGE: int, Gender: str, TYPE_OF_ADMISSION: str, UREA: float, TLC: float, STABLE_ANGINA: str, ACS: str,CREATININE: float, ATYPICAL_CHEST_PAIN: str, ANAEMIA: str, STEMI: str, GLUCOSE: float, HB: float):
    model = app.state.model
    conversion_dict = {
        'GENDER': {'Male': 'M', 'Female': 'F'},
        'TYPE OF ADMISSION-EMERGENCY/OPD': {'EMERGENCY': 'E', 'OPD': 'O'},
        #'UREA':{27},
        #'TLC': {7.8},
        'STABLE_ANGINA': {'Yes' :1 , 'No':0},
        'ACS': {'Yes' :1 , 'No':0},
        #'CREATININE': {1.2},
        'ATYPICAL CHEST PAIN' : {'Yes' :1 , 'No':0},
        'ANAEMIA': {'Yes' :1 , 'No':0},
        'STEMI' : {'Yes' :1 , 'No':0},
        #'GLUCOSE':{120},
        #'HB': {13.0}
    }


    columns = ['AGE', 'GENDER', 'RURAL', 'TYPE_OF_ADMISSION', 'SMOKING ',
       'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD', 'HB', 'TLC',
       'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'RAISED CARDIAC ENZYMES',
       'EF', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI',
       'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR',
       'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
       'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
       'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
       'PULMONARY EMBOLISM', 'CHEST INFECTION']

    # Create dataframe with user input
    df = create_df(conversion_dict, AGE,Gender, TYPE_OF_ADMISSION, UREA, TLC, STABLE_ANGINA, ACS, CREATININE, ATYPICAL_CHEST_PAIN, ANAEMIA, STEMI, GLUCOSE, HB)

    # Preprocessing the input
    X_proc = sk_learn_proc(df)[list(model.feature_names_in_)]
    X_proc.to_csv('abd.csv')

    # Predicting with the model

    y_pred = model.predict(X_proc)

    return {"Prediction": f"{y_pred}"}
