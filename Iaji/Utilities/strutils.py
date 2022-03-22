#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:00:54 2022

@author: jiedz
This module contains utility function for strings.
"""
# In[imports]
import numpy

# In[]
def de_latexify(s):
    """
    From a latex string, generates a python-friendly string
    """
    replacements = [("{", ""), ("}", ""), \
                    ("\\left(", ""), ("\\right)", ""),\
                    ("\\left[", ""), ("\\right]", ""),\
                    ("\\left\\{", ""), ("\\right)=\\}", ""),\
                    ("\\left\\langle", ""), ("\\right\\rangle", ""),\
                    ("\\mathbf", ""), ("\\mathbb", ""),\
                    ("\\hat", ""), ("\\hat", ""),\
                    ("^*", ".conjugate()"),\
                    ("\\tilde", ""),\
                    #("^", "**"),\
                    ("^", "_"),\
                    #(" ", ""),\
                    ("frac", ""),\
                   # ("[", ""), ("]", ""),\
                    ("{", ""), ("}", ""),\
                    ("\\", ""), ("\\", "")]
    s_new = s
    for r in replacements:
        s_new = s_new.replace(*r)
    return s_new

def any_in_string(list, string):
    """
    Checks if any string of a list is contained in the target string
    :param list: iterable of str
        list of substrings
    :param string:
        target string
    :return: bool
        True if any of the strings in the list is contained in the target string
    """
    any = False
    list = numpy.atleast_1d(list)
    for substring in list:
        any = any or substring in string
        if any:
            return True
    return False
            