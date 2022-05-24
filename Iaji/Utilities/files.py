#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:11:40 2022

@author: jiedz
This module contains useful functions to deal with data loading and saving
"""
# In[imports]
import PyQt5
import os
# In[Loading data with a GUI]
def select_directory(start_directory=None, title="Select directory"):
    """
    INPUTS
    -----------
    start_directory : str
        the absolute path of the starting directory.
    """
    if start_directory is None:
        start_directory = os.getcwd()
    #Let the user choose a different data path, if needed.
    #--------------------------------
    directory = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(caption=title, directory=start_directory)
    return directory
