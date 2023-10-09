import logging
from logging.handlers import TimedRotatingFileHandler


class Logger:
    def __init__(self):
        # logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(levelname)s - %(message)s')
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.INFO)
        log_filename = "logs/my_log.log"
        handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=3)
        formatter = logging.Formatter('%(asctime)s: [%(levelname)s]%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)


logger = Logger()
