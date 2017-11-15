import logging
import os

[staticmethod]


def init_logger(log_level=logging.INFO):

    logging.basicConfig(level=log_level)
    log = logging.getLogger('MWDB')
    # create file handler for logging purpose
    log_file = os.getcwd() + "/error.log"
    if os.path.exists(log_file):
        open('info.log', 'w+')
    fh = logging.FileHandler('error.log')
    log.addHandler(fh)
    return log
