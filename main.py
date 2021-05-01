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

# OTHER IMPORTS
import heapq
import os

# NUMPY AND PANDAS
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# MATPLOTLIB
import matplotlib.pyplot as plt
import matplotlib as mpl

# SCIKIT-LEARN IMPORTS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Change background color to white
Window.clearcolor = (1, 1, 1, 0.9)

#Import medical dataset
dataset = pd.read_csv("D:\\OneDrive\\Desktop\\Programming\\Python Scripts\\Kivy\\GradProject\\Datasets\\Dataset_v6.csv",names=['symptom', 'disease'])

#Split into symptoms and diseases
symptoms, diseases = dataset[['symptom']], dataset[['disease']]

#Normalizing the dataset, changing nominal values into numerical values
symptoms_norm = pd.get_dummies(symptoms, drop_first=True)
symptoms_norm.columns = symptoms_norm.columns.str.replace("symptom_", "")
diseases_norm = pd.get_dummies(diseases, drop_first=True)
diseases_norm.columns = diseases_norm.columns.str.replace("disease_", "")

#Create an empy user symptom
symptom_user = np.zeros(shape=len(symptoms_norm.columns)).reshape(1,-1)
probabilities = [None] * len(diseases_norm.columns)

#Abstract window
class Window(Screen):
    answer1 = BooleanProperty(False)
    answer2 = BooleanProperty(False)
    answer3 = BooleanProperty(False)
    answer4 = BooleanProperty(False)
    answer5 = BooleanProperty(False)
    answer6 = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(Window, self).__init__(**kwargs)

    def set_value(self, index, value):
        """Sets the given index to a value"""
        symptom_user[0, index] = value
        print(f"{symptoms_norm.columns[index]} = {symptom_user[0, index]}")

    def print_symptom(self, index):
        return symptoms_norm.columns[index]

    def reset_answer(self):
        self.answer1 = BooleanProperty(False)
        self.answer2 = BooleanProperty(False)
        self.answer3 = BooleanProperty(False)
        self.answer4 = BooleanProperty(False)
        self.answer5 = BooleanProperty(False)
        self.answer6 = BooleanProperty(False)

class FirstWindow(Window):
    pass

class SecondWindow(Window):
    pass

class ThirdWindow(Window):
    pass

class FourthWindow(Window):
    pass

class FifthWindow(Window):
    pass

class SixthWindow(Window):
    pass

class SeventhWindow(Window):
    pass

class EighthWindow(Window):
    pass

class NinthWindow(Window):
    def train(self):
        """Training"""
        # Fit final model
        model = LogisticRegression(solver='lbfgs')
        for disease_prob in range(len(diseases_norm.iloc[0])):
            # Split into train set and test set
            X_train, X_test, y_train, y_test = train_test_split(symptoms_norm, diseases_norm.iloc[:, 0], train_size=0.7,
                                                                test_size=0.3, shuffle=True)
            model.fit(X_train, y_train)
            features = model.predict_proba(symptom_user)
            probabilities[disease_prob] = int(features[0, 1] * 10000)
            print(f"Score for the disease {diseases_norm.columns.values[disease_prob]}:\n{round(features[0, 1] * 10000, 2)}")

        prob_indices = heapq.nlargest(5, range(len(probabilities)), probabilities.__getitem__)
        max_prob = [None] * 5
        max_disease = [None] * 5

        for index in range(len(prob_indices)):
            max_prob[index] = probabilities[prob_indices[index]]
            max_disease[index] = diseases_norm.columns.values[prob_indices[index]]

        # Set font size
        mpl.rcParams['font.size'] = 9.0

        # Colors
        colors = ['#8870FF', '#6DD582', '#D2D56D', '#D57E6D', '#D26DD5']

        fig, ax = plt.subplots()
        ax.pie(max_prob, colors=colors, labels=max_disease, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

        # Draw circle
        centre_circle = plt.Circle((0, 0), 0.85, fc='white')
        fig2 = plt.gcf()
        fig2.gca().add_artist(centre_circle)

        if (os.path.exists('train_results.png')):
            os.remove('train_results.png')
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig('train_results.png')
        plt.show()

class FinalWindow(Window):
    pass

class WindowManager(ScreenManager):
    pass
    

kv = Builder.load_file("dpbs.kv")

#Build
class DPBSApp(App):
    def build(self):
        return kv
    
if __name__ == "__main__":
    DPBSApp().run()