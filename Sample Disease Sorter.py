# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:01:20 2021

@author: alper
"""

import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

#Import medical dataset
dataset = pd.read_csv("Dataset_v6.csv",names=['symptom', 'disease'])
np.set_printoptions(suppress=True)

#Split into symptoms and diseases
symptoms, diseases = dataset[['symptom']], dataset[['disease']]

#Normalizing the dataset, changing nominal values into numerical values
symptoms_norm = pd.get_dummies(symptoms, drop_first=True)
symptoms_norm.columns = symptoms_norm.columns.str.replace("symptom_", "")
diseases_norm = pd.get_dummies(diseases, drop_first=True)
diseases_norm.columns = diseases_norm.columns.str.replace("disease_", "")

X_train, X_test, Y_train, Y_test = train_test_split(symptoms_norm, diseases_norm, train_size= 0.5, test_size= 0.5)

model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))

model.fit(X_train,Y_train)

#Create an empy user symptom
symptom_user = np.zeros(shape=len(symptoms_norm.columns)).reshape(1,-1)

prediction = model.predict(symptom_user)

# Print model score
print(f"Predicted output: {prediction}")