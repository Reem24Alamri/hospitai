from fastapi import FastAPI
from pydantic import BaseModel
from hospital.ml_logic.data import get_data_cleaned
from hospital.ml_logic.preprocessing import sk_learn_proc
from hospital.ml_logic.registry import load_model
from hospital.ml_logic.model import evaluate_model
from sklearn.model_selection import train_test_split
import pickle
app = FastAPI()

def load_model():
    with open('/home/reema/code/Reem24Alamri/hospitai/knn_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model


app.state.model = load_model()


# class Item(BaseModel):
#     Gender: int
#     AGE: int
#     HEART_FAILURE: int
#     AKI: int
#     CVA_INFRACT: int
#     CARDIOGENIC_SHOCK: int
#     PULMONARY_EMBOLISM: int
#     TYPE_OF_ADMISSION: int
#     SMOKING: int
#     DM: int
#     HTN: int
#     SEVERE_ANAEMIA: int

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/testing")
def testing():
    df = get_data_cleaned()
    X, y = sk_learn_proc(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = app.state.model
    model.fit(X_train, y_train)
    mae = evaluate_model(model, X_test, y_test)

    return {"MAE":f"{mae}"}

@app.get("/predict")
def read_item(Gender: str, AGE: int, SMOKING: str, HEART_FAILURE: str,
              AKI: str, CVA_INFRACT: str, CARDIOGENIC_SHOCK: str,
              PULMONARY_EMBOLISM: str, TYPE_OF_ADMISSION: str,
               DM: str, HTN: str, SEVERE_ANAEMIA: str):

    conversion_dict = {
        'SMOKING ': {'YES': 1, 'NO': 0},
        'ALCOHOL': {'YES': 1, 'NO': 0},
        'DM': {'YES': 1, 'NO': 0},
        'HTN': {'YES': 1, 'NO': 0},
        'CAD': {'YES': 1, 'NO': 0},
        'PRIOR_CMP': {'YES': 1, 'NO': 0},
        'CKD': {'YES': 1, 'NO': 0},
        'RAISED_CARDIAC_ENZYMES': {'YES': 1, 'NO': 0},
        'SEVERE_ANAEMIA': {'YES': 1, 'NO': 0},
        'ANAEMIA': {'YES': 1, 'NO': 0},
        'STABLE_ANGINA': {'YES': 1, 'NO': 0},
        'ACS': {'YES': 1, 'NO': 0},
        'STEMI': {'YES': 1, 'NO': 0},
        'ATYPICAL_CHEST_PAIN': {'YES': 1, 'NO': 0},
        'HEART_FAILURE': {'YES': 1, 'NO': 0},
        'HFREF': {'YES': 1, 'NO': 0},
        'HFNEF': {'YES': 1, 'NO': 0},
        'VALVULAR': {'YES': 1, 'NO': 0},
        'CHB': {'YES': 1, 'NO': 0},
        'SSS': {'YES': 1, 'NO': 0},
        'AKI': {'YES': 1, 'NO': 0},
        'CVA_INFRACT': {'YES': 1, 'NO': 0},
        'CVA_BLEED': {'YES': 1, 'NO': 0},
        'AF': {'YES': 1, 'NO': 0},
        'VT': {'YES': 1, 'NO': 0},
        'PSVT': {'YES': 1, 'NO': 0},
        'CONGENITAL': {'YES': 1, 'NO': 0},
        'UTI': {'YES': 1, 'NO': 0},
        'NEURO_ARDIOGENIC_SYNCOPE': {'YES': 1, 'NO': 0},
        'ORTHOSTATIC': {'YES': 1, 'NO': 0},
        'INFECTIVE_ENDOCARDITIS': {'YES': 1, 'NO': 0},
        'DVT': {'YES': 1, 'NO': 0},
        'CARDIOGENIC_SHOCK': {'YES': 1, 'NO': 0},
        'SHOCK': {'YES': 1, 'NO': 0},
        'PULMONARY_EMBOLISM': {'YES': 1, 'NO': 0},
        'CHEST_INFECTION': {'YES': 1, 'NO': 0},
        'GENDER': {'MALE': 'M', 'FEMALE': 'F'},
        'RURAL': {'RURAL': 'R', 'URBAN': 'U'},
        'TYPE_OF_ADMISSION' : {'EMERGENCY':'E', 'OPD': 'O'}
    }
    columns = [
        'SMOKING ', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',
        'RAISED_ARDIAC_ENZYMES', 'SEVERE_ANAEMIA', 'ANAEMIA', 'STABLE_ANGINA',
        'ACS', 'STEMI', 'ATYPICAL_CHEST_PAIN', 'HEART_FAILURE', 'HFREF', 'HFNEF',
        'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA_INFRACT', 'CVA_BLEED', 'AF', 'VT',
        'PSVT', 'CONGENITAL', 'UTI', 'NEURO_CARDIOGENIC_SYNCOPE', 'ORTHOSTATIC',
        'INFECTIVE_ENDOCARDITIS', 'DVT', 'CARDIOGENIC_SHOCK', 'SHOCK',
        'PULMONARY_EMBOLISM', 'CHEST_INFECTION', 'GENDER', 'RURAL','TYPE_OF_ADMISSION'
    ]
    user_input = {}
    for column in columns:
        while True:
            response = input(f"Enter value for {column} (Options: {', '.join(conversion_dict[column].keys())}): ").strip().upper()
            if response in conversion_dict[column]:
                user_input[column] = conversion_dict[column][response]
                break
            else:
                print(f"Invalid input. Please enter one of the following options: {', '.join(conversion_dict[column].keys())}")

    for key, value in user_input.items():
        print(f"{key}: {value}")


    # preprocessing the input
    # X_proc = sk_learn_proc(X)[0]
    # load the model from pickle
    # y_pred = model.predict(X_proc)

    # return {"Prediction": f"{y_pred}"}
