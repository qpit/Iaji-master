#%%
import logging
#%%

class InconsistentShapeError(Exception):
    def __init__(self, error_message="Matrix shapes are mutually inconsistent"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)