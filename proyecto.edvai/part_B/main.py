from fastapi import FastAPI
import uvicorn
import requests

from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder

import os
app = FastAPI()


MAIN_FOLDER = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MAIN_FOLDER, "modelo_proyecto_final.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

COLUMNS_PATH = os.path.join(MAIN_FOLDER,"part_B\categories_ohe_without_fraudulent.pickle")
with open(COLUMNS_PATH, 'rb') as handle:
        ohe_tr = pickle.load(handle)


BINS_ORDER = os.path.join(MAIN_FOLDER, "saved_bins_order.pickle")
with open(BINS_ORDER, 'rb') as handle:
        new_saved_bins_order = pickle.load(handle)

BINS_TRANSACTION = os.path.join(MAIN_FOLDER, "saved_bins_transaction.pickle")
with open(BINS_TRANSACTION, 'rb') as handle:
        new_saved_bins_transaction = pickle.load(handle)

    




class Answer(BaseModel):
    orderAmount: int
    orderState: str 
    paymentMethodRegistrationFailure: bool 
    paymentMethodType: str 
    paymentMethodProvider: str 
    paymentMethodIssuer: str 
    transactionAmount: int
    transactionFailed: bool 
    emailDomain: str 
    emailProvider: str 
    customerIPAddressSimplified: str 
    sameCity: str 


@app.get("/")
async def root():
    return {"message": "Proyecto para Bootcamp de EDVAI "}

@app.post("/prediccion")
def predict_fraud_customer(answer: Answer):

    answer_dict = jsonable_encoder(answer)
    
    for key, value in answer_dict.items():
        answer_dict[key] = [value]





    single_instance = pd.DataFrame.from_dict(answer_dict)

    single_instance["orderAmount"] = single_instance["orderAmount"].astype(float)
    single_instance["orderAmount"] = pd.cut(single_instance['orderAmount'],
                                    bins=new_saved_bins_order, 
                                    include_lowest=True)
    
    single_instance["transactionAmount"] = single_instance["transactionAmount"].astype(int)
    single_instance["transactionAmount"] = pd.cut(single_instance['transactionAmount'],
                                     bins=new_saved_bins_order, 
                                     include_lowest=True)
    
   
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)

    prediction = model.predict(single_instance_ohe)






 # Cast numpy.int64 to just a int
    type_of_fraud = int(prediction[0])
        
    response = {"Tipo de fraude": type_of_fraud}

# Corre en http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)