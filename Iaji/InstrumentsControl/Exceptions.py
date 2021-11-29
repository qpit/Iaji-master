"""
This file describes common exceptions 
"""
#%%
class ConnectionError(Exception):
    def __init__(self, error_message="Could not connect to instrument"):
        print(error_message)