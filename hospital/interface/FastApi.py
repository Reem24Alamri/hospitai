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
    if Gender == 'M':
        Gender = 'Male'
    else:
        Gender = 'Female'

    if SMOKING == 'Yes':
        SMOKING = '1'
    else:
        SMOKING = '0'

    if  TYPE_OF_ADMISSION == 'Emergency ':
        TYPE_OF_ADMISSION = 'E'
    else:
        TYPE_OF_ADMISSION = 'O'

    if  HEART_FAILURE == 'Yes':
        HEART_FAILURE = '1'
    else:
        HEART_FAILURE = '0'

    if  AKI  == 'Yes':
        AKI  = '1'
    else:
        AKI  = '0'
    if  CVA_INFRACT  == 'Yes':
        CVA_INFRACT  = '1'
    else:
        CVA_INFRACT  = '0'

    if  CARDIOGENIC_SHOCK  == 'Yes':
        CARDIOGENIC_SHOCK  = '1'
    else:
        CARDIOGENIC_SHOCK  = '0'

    if  PULMONARY_EMBOLISM == 'Yes':
        PULMONARY_EMBOLISM = '1'
    else:
        PULMONARY_EMBOLISM = '0'

    if  DM == 'Yes':
        DM = '1'
    else:
        DM = '0'

    if  HTN == 'Yes':
        HTN = '1'
    else:
        HTN = '0'

    if  SEVERE_ANAEMIA == 'Yes':
        SEVERE_ANAEMIA = '1'
    else:
        SEVERE_ANAEMIA = '0'



    summary = (
        f"Gender: {Gender}, AGE: {AGE}, SMOKING: {SMOKING}, HEART_FAILURE: {HEART_FAILURE}, "
        f"AKI: {AKI}, CVA_INFRACT: {CVA_INFRACT}, CARDIOGENIC_SHOCK: {CARDIOGENIC_SHOCK}, "
        f"PULMONARY_EMBOLISM: {PULMONARY_EMBOLISM}, TYPE_OF_ADMISSION: {TYPE_OF_ADMISSION}, "
        f"DM: {DM}, HTN: {HTN}, SEVERE_ANAEMIA: {SEVERE_ANAEMIA}"
    )

    # preprocessing the input
    # X_proc = sk_learn_proc(X)[0]
    # load the model from pickle
    # y_pred = model.predict(X_proc)

    # return {"Prediction": f"{y_pred}"}
