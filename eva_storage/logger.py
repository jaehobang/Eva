"""
This file implements the logging manager that is used to create / print logs throughout all the program

"""

import logging



class Logger:
    _instance = None
    _LOG = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)

            # LOGGING INITIALIZATION
            cls._LOG = logging.getLogger(__name__)
            LOG_handler = logging.StreamHandler()
            LOG_formatter = logging.Formatter(
                fmt='%(asctime)s [%(funcName)s:%(lineno)03d]'
                    '%(levelname)-5s: %(message)s',
                datefmt='%m-%d-%Y %H:%M:%S'
            )
            LOG_handler.setFormatter(LOG_formatter)
            cls._LOG.addHandler(LOG_handler)
            cls._LOG.setLevel(logging.INFO)

        return cls._instance



    def error(self, message):
        self._LOG.error(message)

    def log(self, message):
        self._LOG.info(message)



