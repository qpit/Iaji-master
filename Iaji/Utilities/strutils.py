#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:00:54 2022

@author: jiedz
This module contains utility function for strings.
"""
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
            