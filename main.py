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

# SCIKIT LEARN IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# IMPORTS
import numpy as np
import pandas as pd

#Change background color to white
Window.clearcolor = (1, 1, 1, 0.9)

class MainWindow(Screen):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)

class SecondWindow(Screen):
    def __init__(self, **kwargs):
        super(SecondWindow, self).__init__(**kwargs)
        
    answer1 = BooleanProperty(False)
    answer2 = BooleanProperty(False)
    answer3 = BooleanProperty(False)

class ThirdWindow(Screen):
    def __init__(self, **kwargs):
        super(ThirdWindow, self).__init__(**kwargs)
        
    

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