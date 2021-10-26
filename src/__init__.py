# -*- coding: utf-8 -*-

import logging
import copy
from logging import FileHandler, StreamHandler

debug = False
logger = None


class Struct:
    """
        Class whose instances attributes are constructed at runtime
        from a given dictionary 'entries'. Useful as a container of
        sets of parameters, avoiding to access them via the dictionary
        interface.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

def setup_logger(filename, debug=False):

    logging_level = logging.INFO
    if debug :
        wbsp.debug = debug
        logging_level = logging.DEBUG

    log_filename = '{}'.format(filename)
    print("Logging to file: {}".format(log_filename))
    #logging.basicConfig( #filename = log_filename, \
    #                    level = logging_level,\
    #                    format='%(asctime)s [%(levelname)s] - %(message)s' )
    wbsp.logger = logging.getLogger(__name__)
    wbsp.logger.setLevel(logging_level)
    wbsp.logger.addHandler( FileHandler(log_filename, mode='w') )
    wbsp.logger.addHandler( StreamHandler() )
    for handler in wbsp.logger.handlers :
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))
