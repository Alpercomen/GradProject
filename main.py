# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:43:41 2021

@author: alper
"""

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window

#Change background color to white
Window.clearcolor = (1, 1, 1, 0.9)

class MainWindow(Screen):
    pass

class SecondWindow(Screen):
    pass

class ThirdWindow(Screen):
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