"""
This module describes common error classes.
"""
#%%
class ConnectionError(Exception):
    def __init__(self, error_message="Could not connect"):
        print(error_message)
#%%
class ResonanceNotFoundError(Exception):
    def __init__(self, error_message="Could not connect"):
        print(error_message)