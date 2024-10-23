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

default_values = {
        'ALCOHOL': 0,
        'CAD': 1,
        'PRIOR CMP': 0,
        'CKD': 0,
        'RAISED CARDIAC ENZYMES': 0,
        'ANAEMIA': 0,
        'STABLE ANGINA': 0,
        'ACS': 0,
        'STEMI': 0,
        'ATYPICAL CHEST PAIN': 0,
        'HFREF': 0,
        'HB': 12.6,
        'TLC': 8.4,
        'PLATELETS': 150,
        'GLUCOSE': 110,
        'UREA': 27,
        'CREATININE': 0.8,
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
        'PULMONARY EMBOLISM': 0,
        'PRIOR CMP': 0,
        'STABLE ANGINA': 0,
        'HEART FAILURE': 1,
        'RAISED CARDIAC ENZYMES': 1,
        'CVA INFRACT': 0,
        'NEURO CARDIOGENIC SYNCOPE': 0
    }

def create_df(conversion_dict, age, gender, smoking, heart_failure, aki, cva_infract, pulmonary_embolism, type_of_admission, dm, htn, cardiogenic_shock, severe_anaemia):
    df_dict = {'AGE':age}
    features = [
        gender, smoking, heart_failure, aki, cva_infract, pulmonary_embolism, type_of_admission, dm,htn, cardiogenic_shock, severe_anaemia
    ]

    for k, f in zip(conversion_dict.keys(), features):
        df_dict[k] = [conversion_dict[k][f]]

    combined_dict = df_dict | default_values

    # Create the DataFrame
    return pd.DataFrame(combined_dict)


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/testing")
def testing():

    # df = get_data_cleaned()
    # X, y = sk_learn_proc(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model.fit(X_train, y_train)
    # mae = evaluate_model(model, X_test, y_test)

    model = app.state.model

    conversion_dict = {
        'GENDER': {'MALE': 'M', 'FEMALE': 'F'},
        'SMOKING ': {'YES': 1, 'NO': 0},
        'HEART FAILURE': {'YES': 1, 'NO': 0},
        'AKI': {'YES': 1, 'NO': 0},
        'CVA INFRACT': {'YES': 1, 'NO': 0},
        'PULMONARY EMBOLISM': {'YES': 1, 'NO': 0},
        'TYPE OF ADMISSION-EMERGENCY/OPD' : {'EMERGENCY':'E', 'OPD': 'O'},
        'DM': {'YES': 1, 'NO': 0},
        'HTN': {'YES': 1, 'NO': 0},
        'CARDIOGENIC SHOCK': {'YES': 1, 'NO': 0},
        'SEVERE ANAEMIA': {'YES': 1, 'NO': 0}
    }

    new_df = create_df(
        conversion_dict, 30, 'MALE', 'YES','YES','YES','YES','YES','EMERGENCY','YES','YES','YES','YES'
    )

    proc_new = sk_learn_proc(new_df)

    return {"Prediction":f"{model.predict(sk_learn_proc(new_df))[0]}"}

@app.get("/predict")
# def read_item(Gender: str, AGE: int, SMOKING: str, HEART_FAILURE: str,
#               AKI: str, CVA_INFRACT: str, CARDIOGENIC_SHOCK: str,
#               PULMONARY_EMBOLISM: str, TYPE_OF_ADMISSION: str,
#                DM: str, HTN: str, SEVERE_ANAEMIA: str):
def read_item():

    conversion_dict = {
        'GENDER': {'MALE': 'M', 'FEMALE': 'F'},
        'SMOKING ': {'YES': 1, 'NO': 0},
        'HEART FAILURE': {'YES': 1, 'NO': 0},
        'AKI': {'YES': 1, 'NO': 0},
        'CVA INFRACT': {'YES': 1, 'NO': 0},
        'PULMONARY EMBOLISM': {'YES': 1, 'NO': 0},
        'TYPE OF ADMISSION-EMERGENCY/OPD' : {'EMERGENCY':'E', 'OPD': 'O'},
        'DM': {'YES': 1, 'NO': 0},
        'HTN': {'YES': 1, 'NO': 0},
        'CARDIOGENIC SHOCK': {'YES': 1, 'NO': 0},
        'SEVERE ANAEMIA': {'YES': 1, 'NO': 0}
    }
    # columns = ['AGE', 'GENDER', 'SMOKING ', 'HEART FAILURE', 'AKI', 'CVA INFRACT',
    #    'PULMONARY EMBOLISM', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'DM', 'HTN',
    #    'CARDIOGENIC SHOCK', 'SEVERE ANAEMIA', 'ALCOHOL', 'CAD', 'PRIOR CMP',
    #    'CKD', 'RAISED CARDIAC ENZYMES', 'ANAEMIA', 'STABLE ANGINA', 'ACS',
    #    'STEMI', 'ATYPICAL CHEST PAIN', 'HFREF', 'HB', 'TLC', 'PLATELETS',
    #    'GLUCOSE', 'UREA', 'CREATININE', 'EF', 'RURAL', 'HFNEF', 'VALVULAR',
    #    'CHB', 'SSS', 'CVA BLEED', 'AF', 'VT', 'PSVT', 'CONGENITAL', 'UTI',
    #    'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC', 'INFECTIVE ENDOCARDITIS',
    #    'DVT', 'SHOCK', 'CHEST INFECTION']
    columns = ['AGE', 'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'SMOKING ',
       'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD', 'HB', 'TLC',
       'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'RAISED CARDIAC ENZYMES',
       'EF', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA', 'ACS', 'STEMI',
       'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF', 'VALVULAR',
       'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
       'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
       'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
       'PULMONARY EMBOLISM', 'CHEST INFECTION']


    df = pd.DataFrame()
    # user_input = {}
    # for column in columns:
    #     while True:
    #         response = input(f"Enter value for {column} (Options: {', '.join(conversion_dict[column].keys())}): ").strip().upper()
    #         if response in conversion_dict[column]:
    #             user_input[column] = conversion_dict[column][response]
    #             break
    #         else:
    #             print(f"Invalid input. Please enter one of the following options: {', '.join(conversion_dict[column].keys())}")

    # for key, value in user_input.items():
    #     print(f"{key}: {value}")

    additional_values = {
        'ALCOHOL': 0,
        'CAD': 1,
        'PRIOR CMP': 0,
        'CKD': 0,
        'RAISED CARDIAC ENZYMES': 0,
        'ANAEMIA': 0,
        'STABLE ANGINA': 0,
        'ACS': 0,
        'STEMI': 0,
        'ATYPICAL CHEST PAIN': 0,
        'HFREF': 0,
        'HB': 12.6,
        'TLC': 8.4,
        'PLATELETS': 150,
        'GLUCOSE': 110,
        'UREA': 27,
        'CREATININE': 0.8,
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
        'PULMONARY EMBOLISM': 0,
        'PRIOR CMP': 0,
        'STABLE ANGINA': 0,
        'HEART FAILURE': 1,
        'RAISED CARDIAC ENZYMES': 1,
        'CVA INFRACT': 0,
        'NEURO CARDIOGENIC SYNCOPE': 0
    }


    # preprocessing the input
    # X_proc = sk_learn_proc(X)[0]
    # load the model from pickle
    # y_pred = model.predict(X_proc)

    # return {"Prediction": f"{y_pred}"}
