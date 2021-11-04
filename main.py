# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:43:41 2021

"""
# collections
# KIVY IMPORTS
import collections

from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.properties import BooleanProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label

# OTHER IMPORTS
import operator
import itertools

# NUMPY AND PANDAS
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# PIECHART - Special thanks to MrTequila on https://github.com/MrTequila/kivy-PieChart
import PieChart as pc

# SCIKIT-LEARN IMPORTS
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Preliminary setups
Window.clearcolor = (1, 1, 1, 0.9)
np.set_printoptions(suppress=True) #This prevents showing scientific notation.

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
symptom_user = np.zeros(shape=len(symptoms_norm.columns)).reshape(1, -1)

#Abstract window
class Window(Screen):
    #Checkbox results in one window are stored here, initially False
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
        #Non-iterable, reseting answers
        self.answer1.active = False
        self.answer2.active = False
        self.answer3.active = False
        self.answer4.active = False
        self.answer5.active = False
        self.answer6.active = False

    def reset_answer2(self):
        #Non-iterable, reseting answers for the last page only.
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
    grid = None # To be able to show the results in UI, an empty grid is set.
    model = None # To avoid training over and over again everytime input is entered, training model is stored here.
    def train(self):
        """Training"""
        #Store train id
        train_id = self.ids.train_results
        self.grid = GridLayout(cols=1, spacing='0dp')

        global X_train, X_test, Y_train, Y_test

        if self.model == None:
            #Divide the model into train and test sets
            X_train, X_test, Y_train, Y_test = train_test_split(symptoms_norm, diseases_norm, test_size= 0.165)

            #Create model
            self.model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))

            #Fit Model
            self.model.fit(X_train,Y_train)

        if np.all(symptom_user == 0):
            self.label = Label(text="[color=#000000]You have entered no symptoms![/color]", font_size= "20sp", markup = True)
            self.grid.add_widget(self.label)
            train_id.add_widget(self.grid)
            return

        #Make predictions
        prediction = self.model.predict(symptom_user)

        # Print and store model score
        model_scores = [None] * len(Y_test.columns)
        print(f"Predicted output: {prediction[0]}")
        for i in range(len(Y_test.columns)):
            model_scores[i] = mean_squared_error(Y_test.iloc[:, i], prediction[0])
            print(f"{Y_test.columns[i]} MSE score: {model_scores[i]}")


        keys = [None] * len(diseases_norm.columns)
        values = [None] * len(diseases_norm.columns)
        for i in range(len(prediction[0])):
            keys[i] = diseases_norm.columns.values[i]
            values[i] = int(prediction[0][i]*1000000)
            print(f"'{keys[i]}': {values[i]},")

        #Create a dictionary containing diseases and its respective percentage
        zip_iterator = zip(keys, values)
        in_data = dict(zip_iterator)

        #Sort it in reverse order
        sorted_dict = dict(sorted(  in_data.items(),
                            key=operator.itemgetter(1),
                            reverse=True))

        #Take the first five values, resulting in five largest values
        n_item = dict(itertools.islice(sorted_dict.items(), 0, 5))

        #Create and send a Pie Chart to front end
        position = (250, 250)
        size = (250, 250)
        chart = pc.PieChart(data=n_item, position=position, size=size, legend_enable=True)
        self.grid = GridLayout(cols=1, spacing='0dp')
        self.grid.add_widget(chart)
        train_id.add_widget(self.grid)

    def removeChart(self):
        """"Remoes Chart from front-end"""
        train_id = self.ids.train_results
        train_id.remove_widget(self.grid)

    def resetSymptom(self):
        """Resets the given symptom input by the user"""
        global symptom_user
        symptom_user = np.zeros(shape=len(symptoms_norm.columns)).reshape(1, -1)

class WindowManager(ScreenManager):

    #ScreenManager.screens is Non-Iterable, therefore we have no choice but to copy+paste here
    def reset_windows(self):
        """"Resets each UI check box"""
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