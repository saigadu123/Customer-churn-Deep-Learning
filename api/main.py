from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle 

app = FastAPI()

model = load_model('./model/bank_churn_ann.h5')
scaler = pickle.load(open('./model/scaler.pkl','rb'))

class CustomerFeatures(BaseModel):
    CreditScore: int 
    Geography : str 
    Gender : str 
    Age : int 
    Tenure: int
    Balance: float
    NumOfProducts: int 
    HasCrCard: int 
    IsActiveMember: int
    EstimatedSalary: float 

geo_map = {'France':0,'Spain':1,'Germany':2}
gender_map = {'Female':0,'Male':1}

@app.post('/predict')
def predict_churn(features: CustomerFeatures):

    data = features.dict()

    data['Geography'] = geo_map.get(data['Geography'],0)
    data['Gender'] = gender_map.get(data['Gender'],0)

    x = pd.DataFrame(data, index=[0])

    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0][0]
    return {'churn_probability':float(pred),'churned': int(pred>0.5)}
