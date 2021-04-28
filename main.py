# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:43:41 2021

@author: alper
"""

# KIVY IMPORTS
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.properties import BooleanProperty

#Change background color to white
Window.clearcolor = (1, 1, 1, 0.9)

# NUMPY AND PANDAS
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# SCIKIT-LEARN IMPORTS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Create an empy user symptom
symptom_user = np.zeros(shape=32).reshape(1,-1)

class MainWindow(Screen):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)

class SecondWindow(Screen):
    answer1 = BooleanProperty(False)
    answer2 = BooleanProperty(False)
    answer3 = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super(SecondWindow, self).__init__(**kwargs)
        
    def set_value(self, index, value):
        """Sets the given index to a value"""
        symptom_user[0, index] = value
        print(f"Symptom[{index}] = {symptom_user[0, index]}")
    
    

class ThirdWindow(Screen):
    answer4 = BooleanProperty(False)
    answer5 = BooleanProperty(False)
    answer6 = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super(ThirdWindow, self).__init__(**kwargs)
        
    def set_value(self, index, value):
        """Sets the given index to a value"""
        symptom_user[0, index] = value
        print(f"Symptom[{index}] = {symptom_user[0, index]}")
    
    def train(self):
        """Training"""
        #Import medical dataset
        dataset = pd.read_csv("D:\\OneDrive\\Desktop\\Programming\\Python Scripts\\Kivy\\GradProject\\Datasets\\Dataset_v4.csv",names=['symptom', 'disease'])
        
        #Split into symptoms and diseases
        symptoms, diseases = dataset[['symptom']], dataset[['disease']]
        
        #Normalizing the dataset, changing nominal values into numerical values
        symptoms_norm = pd.get_dummies(symptoms, drop_first=True)
        diseases_norm = pd.get_dummies(diseases, drop_first=True)
        
        #Fit final model
        model = LogisticRegression(solver='lbfgs')
            
        for disease_prob in range(len(diseases_norm.iloc[0])):
            #Split into train set and test set
            X_train, X_test, y_train, y_test = train_test_split(symptoms_norm, diseases_norm.iloc[:,0], train_size= 0.7, test_size=0.3, shuffle=True)
            model.fit(X_train, y_train)
            features = model.predict_proba(symptom_user)
            print(f"Probability for the {diseases_norm.columns.values[disease_prob]}:\n {features}\n\n")   
        
        
class FinalWindow(Screen):
    def __init__(self, **kwargs):
        super(FinalWindow, self).__init__(**kwargs)    
    

class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)
        
    

kv = Builder.load_file("dpbs.kv")

#Build
class DPBSApp(App):
    def build(self):
        return kv
    
if __name__ == "__main__":
    DPBSApp().run()