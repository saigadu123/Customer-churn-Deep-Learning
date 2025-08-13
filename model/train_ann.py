import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

df = pd.read_csv('../data/bank_churn_data.csv')

x = df.drop(['CustomerId','Surname','Exited'],axis=1)
y = df["Exited"]

x['Geography'] = LabelEncoder().fit_transform(x['Geography'])
x['Gender'] = LabelEncoder().fit_transform(x['Gender'])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

model = Sequential([
    Dense(16,activation='relu',input_shape=(x_train.shape[1],)),
    Dense(8,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=30,batch_size=16,validation_split=0.1,verbose=1)

model.save('../model/bank_churn_ann.h5')
pd.to_pickle(scaler,'../model/scaler.pkl')