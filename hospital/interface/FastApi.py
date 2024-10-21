from fastapi import FastAPI
from pydantic import BaseModel
from hospital.ml_logic.preprocessing import sk_learn_proc

app = FastAPI()

class Item(BaseModel):
    Gender: int
    AGE: int
    HEART_FAILURE: int
    AKI: int
    CVA_INFRACT: int
    CARDIOGENIC_SHOCK: int
    PULMONARY_EMBOLISM: int
    TYPE_OF_ADMISSION: int
    SMOKING: int
    DM: int
    HTN: int
    SEVERE_ANAEMIA: int

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/predict")

def read_item(Gender: str, AGE: int, SMOKING: str, HEART_FAILURE: str,
              AKI: str, CVA_INFRACT: str, CARDIOGENIC_SHOCK: str,
              PULMONARY_EMBOLISM: str, TYPE_OF_ADMISSION: str,
               DM: str, HTN: str, SEVERE_ANAEMIA: str):

    Gender = 'Male' if Gender == 'M' else 'Female'

    SMOKING = 'Smoking' if SMOKING == 'Yes' else 'Non smoking'

    TYPE_OF_ADMISSION = 'Yes' if TYPE_OF_ADMISSION == 'Emergency Department' else 'OPD'

    HEART_FAILURE = 'Yes' if HEART_FAILURE == 'Yes' else 'No'

    AKI = 'Yes' if AKI == 'Yes' else 'No'

    CVA_INFRACT = 'Yes' if CVA_INFRACT == 'Yes' else 'No'

    CARDIOGENIC_SHOCK = 'Yes' if CARDIOGENIC_SHOCK == 'Yes' else 'No'

    PULMONARY_EMBOLISM = 'Yes' if PULMONARY_EMBOLISM == 'Yes' else 'No'

    DM = 'Yes' if DM == 'Yes' else 'No'

    HTN = 'Yes' if HTN == 'Yes' else 'No'

    SEVERE_ANAEMIA = 'Yes' if SEVERE_ANAEMIA == 'Yes' else 'No'


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


@app.post("/read-item/")
def api_read_item(item: Item):
    return read_item(
        item.Gender, item.AGE, item.SMOKING, item.HEART_FAILURE,
        item.AKI, item.CVA_INFRACT, item.CARDIOGENIC_SHOCK,
        item.PULMONARY_EMBOLISM, item.TYPE_OF_ADMISSION,
         item.DM, item.HTN, item.SEVERE_ANAEMIA
    )
