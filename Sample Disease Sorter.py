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
dataset=pd.read_csv("D:\\OneDrive\\Desktop\\Programming\\Python Scripts\\Kivy\\GradProject\\Datasets\\sydi.csv",names=['sid','scui','symptom','did','dcui','dicpc','disease','d1','dp','acute'])
dataset['d1'] = dataset['d1'].fillna(0)
dataset['dp'] = dataset['dp'].fillna(0)
dataset['acute'] = dataset['acute'].fillna(0)

#Select features and target
features = ['sid', 'd1', 'dp', 'acute']

X = dataset[features]
y = dataset['did']

#Split into train set and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size= 0.7, test_size=0.3)


#Fit final model
model = LogisticRegression(solver='sag', max_iter=100)
model.fit(X, y)

disease = model.predict_proba(X_val)

print(disease)

