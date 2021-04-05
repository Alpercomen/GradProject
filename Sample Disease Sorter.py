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
dataset = pd.read_csv("D:\\OneDrive\\Desktop\\Programming\\Python Scripts\\Kivy\\GradProject\\Datasets\\sydi.csv",names=['sid','scui','symptom','did','dcui','dicpc','disease','d1','dp','acute'])
dataset['d1'] = dataset['d1'].fillna(0)
dataset['dp'] = dataset['dp'].fillna(0)
dataset['acute'] = dataset['acute'].fillna(0)

#Split into symptoms and diseases
symptoms, diseases = dataset[['sid', 'symptom', 'd1', 'dp', 'acute']], dataset[['did', 'disease']]

#Normalizing the dataset, changing nominal values into numerical values
symptoms_norm = pd.get_dummies(symptoms, drop_first=True)
diseases_norm = pd.get_dummies(diseases, drop_first=True)

#Split into train set and test set
X_train, X_test, y_train, y_test = train_test_split(symptoms_norm, diseases_norm['disease_Abdominal swelling'], train_size= 0.7, test_size=0.3, shuffle=True)

#Fit final model
model = LogisticRegression(solver='saga', max_iter=100000)
model.fit(X_train, y_train)

print(model.predict_proba(X_test))

#Score the model
model.score(X_test, y_test)

#Model Predictions
#model.predict_proba(X_test)[0]