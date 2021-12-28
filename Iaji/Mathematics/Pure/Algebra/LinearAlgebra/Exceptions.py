#%%
import logging
#%%

class InconsistentShapeError(Exception):
    def __init__(self, error_message="Matrix shapes are mutually inconsistent"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)

class TestFailedError(Exception):
    def __init__(self, error_message="Test not conclusive"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)

class TestFailedWarning(Exception):
    def __init__(self, warning_message="Test not conclusive"):
        logging.basicConfig(level=logging.WARNING)
        logging.warning(warning_message)