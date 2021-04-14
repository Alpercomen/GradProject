# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:01:20 2021

@author: alper
"""

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Import medical dataset
dataset = pd.read_csv("D:\\OneDrive\\Desktop\\Programming\\Python Scripts\\Kivy\\GradProject\\Datasets\\Dataset_v4.csv",names=['symptom', 'disease'])

#Split into symptoms and diseases
symptoms, diseases = dataset[['symptom']], dataset[['disease']]

#Normalizing the dataset, changing nominal values into numerical values
symptoms_norm = pd.get_dummies(symptoms, drop_first=True)
diseases_norm = pd.get_dummies(diseases, drop_first=True)

#Fit final model
model = LogisticRegression(solver='lbfgs')

#Create and store user symptoms
symptom_user = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

for x in range(len(diseases_norm.iloc[0])):
    #Split into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(symptoms_norm, diseases_norm.iloc[:,0], train_size= 0.7, test_size=0.3, shuffle=True)
    model.fit(X_train, y_train)
    features = model.predict_proba(symptom_user)
    print(f"Probability for the {diseases_norm.columns.values[x]}:\n {features}\n\n")
