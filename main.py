# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:43:41 2021

@author: alper
"""

# KIVY IMPORTS
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.properties import BooleanProperty
from kivy.uix.gridlayout import GridLayout


# OTHER IMPORTS
import heapq

# NUMPY AND PANDAS
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# PIECHART - Special thanks to MrTequila on https://github.com/MrTequila/kivy-PieChart
import PieChart as pc

# SCIKIT-LEARN IMPORTS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Change background color to white
Window.clearcolor = (1, 1, 1, 0.9)

#Import medical dataset
dataset = pd.read_csv("Dataset_v6.csv",names=['symptom', 'disease'])

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
        self.answer1.active = False
        self.answer2.active = False
        self.answer3.active = False
        self.answer4.active = False
        self.answer5.active = False
        self.answer6.active = False

    def reset_answer2(self):
        self.answer1.active = False
        self.answer2.active = False
        self.answer3.active = False
        self.answer4.active = False

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
    pass

class FinalWindow(Window):
    grid = None
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

            #Print model score
            print(f"Score for the disease {diseases_norm.columns.values[disease_prob]}:\n{round(features[0, 1] * 10000, 2)}")

        # Returns a list of n largest numbers' indices
        prob_indices = heapq.nlargest(5, range(len(probabilities)), probabilities.__getitem__)
        max_prob = [None] * 5
        max_disease = [None] * 5

        for index in range(len(prob_indices)):
            max_prob[index] = probabilities[prob_indices[index]] * (10 * (len(prob_indices) - index))
            max_disease[index] = diseases_norm.columns.values[prob_indices[index]]

        zip_iterator = zip(max_disease, max_prob)
        in_data = dict(zip_iterator)

        position = (250, 250)
        size = (250, 250)
        chart = pc.PieChart(data=in_data, position=position, size=size, legend_enable=True)

        train_id = self.ids.train_results
        self.grid = GridLayout(cols=1, spacing='0dp')
        self.grid.add_widget(chart)
        train_id.add_widget(self.grid)

    def removeChart(self):
        train_id = self.ids.train_results
        train_id.remove_widget(self.grid)

class WindowManager(ScreenManager):

    #ScreenManager.screens is Non-Iterable, therefore we have no choice but to copy+paste here
    def reset_windows(self):
        ScreenManager.get_screen(self, name='second').reset_answer()
        ScreenManager.get_screen(self, name='third').reset_answer()
        ScreenManager.get_screen(self, name='fourth').reset_answer()
        ScreenManager.get_screen(self, name='fifth').reset_answer()
        ScreenManager.get_screen(self, name='sixth').reset_answer()
        ScreenManager.get_screen(self, name='seventh').reset_answer()
        ScreenManager.get_screen(self, name='eighth').reset_answer()
        ScreenManager.get_screen(self, name='ninth').reset_answer2()


kv = Builder.load_file("dpbs.kv")

#Build
class DPBSApp(MDApp):
    def build(self):
        return kv
    
if __name__ == "__main__":
    DPBSApp().run()