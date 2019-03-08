import logging

import progressbar
from config import settings


def setup_logging():
    level = settings.get_setting('Logging', 'level')
    filter_duplicate_messages = settings.get_setting(
        'Logging', 'filter_duplicate_messages', func=bool)

    progressbar.streams.wrap_stderr()
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(filename)s:%(funcName)s: %(message)s')

    if filter_duplicate_messages:
        class DuplicateFilter(logging.Filter):

            def filter(self, record):
                current_log = (record.module, record.levelno, record.msg)
                if current_log != getattr(self, "last_log", None):
                    self.last_log = current_log
                    return True
                return False

        logger = logging.getLogger()
        logger.addFilter(DuplicateFilter())


def setup():
    setup_logging()
