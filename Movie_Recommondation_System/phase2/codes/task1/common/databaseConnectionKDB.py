from qpython import qconnection
import logger

[staticmethod]


def connect_to_kdbdb(host="localhost",port="10000"):
    # Create log handler
    log_handler = logger.init_logger()

    # Connection to database and exception handling
    try:
        q = qconnection.QConnection(host="localhost", port=10000)
        q.open()
        return q
    except qconnection.QConnectionException as err:
        log_handler.error(err)
    except:
        log_handler.error("Unable to connect KDB")

