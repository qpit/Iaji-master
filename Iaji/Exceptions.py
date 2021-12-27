"""
This module defines exceptions for the Matrix module
"""
import logging

class InconsistentArgumentsError(Exception):
    def __init__(self, error_message="Some input arguments are mutually inconsistent"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)


class MissingArgumentsError(Exception):
    def __init__(self, error_message="Too few input arguments"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)

class InvalidArgumentError(Exception):
    def __init__(self, error_message="One input argument is not valid"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)

class InconsistentShapeError(Exception):
    def __init__(self, error_message="Some of the properties' shapes are mutually inconsistent"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)

class MethodNotImplementedError(Exception):
    def __init__(self, error_message="This method has not been implemented yet"):
        logging.basicConfig(level=logging.ERROR)
        logging.error(error_message)


