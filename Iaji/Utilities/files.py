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
def select_data_path(start_directory=None):
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
    file_dialog_title = "Select the directory where data are located"
    data_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(caption=file_dialog_title, directory=start_directory)
    return data_path
