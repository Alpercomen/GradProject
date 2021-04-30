# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:01:20 2021

@author: alper
"""

import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Import medical dataset
dataset = pd.read_csv("D:\\OneDrive\\Desktop\\Programming\\Python Scripts\\Kivy\\GradProject\\Datasets\\Dataset_v5.csv",names=['symptom', 'disease'])

#Split into symptoms and diseases
symptoms, diseases = dataset[['symptom']], dataset[['disease']]

#Normalizing the dataset, changing nominal values into numerical values
symptoms_norm = pd.get_dummies(symptoms, drop_first=True)
symptoms_norm.columns = symptoms_norm.columns.str.replace("symptom_", "")
diseases_norm = pd.get_dummies(diseases, drop_first=True)
diseases_norm.columns = diseases_norm.columns.str.replace("disease_", "")

#Fit final model
model = LogisticRegression(solver='lbfgs')

#Create and store user symptoms
symptom_user = np.zeros(shape=len(symptoms_norm.columns)).reshape(1,-1)

probabilities = [None] * len(diseases_norm.columns)

for disease_prob in range(len(diseases_norm.iloc[0])):
    # Split into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(symptoms_norm, diseases_norm.iloc[:, 0], train_size=0.7,
                                                        test_size=0.3, shuffle=True)
    model.fit(X_train, y_train)
    features = model.predict_proba(symptom_user)
    probabilities[disease_prob] = int(features[0, 1] * 100)
    print(f"Probability for the {diseases_norm.columns.values[disease_prob]}:\n{round(features[0, 1] * 100, 2)}%\n\n")

prob_indices = heapq.nlargest(5, range(len(probabilities)), probabilities.__getitem__)
max_prob = [None] * 5
max_disease = [None] * 5

for index in range(len(prob_indices)):
    max_prob[index] = probabilities[prob_indices[index]]
    max_disease[index] = diseases_norm.columns.values[prob_indices[index]]

plt.pie(max_prob, labels=max_disease, autopct='%1.1f%%')
plt.title('Probabilities')
plt.axis('equal')
plt.show()