"""
This module defines exceptions for the Matrix module
"""
import logging

class UnmatchingArgumentsError(Exception):
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
