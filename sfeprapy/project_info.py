import logging

logger = logging.getLogger('sfeprapy')
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(
    logging.Formatter(fmt='{asctime} {levelname:8.8s} [{filename:15.15s}:{lineno:05d}] {message:s}', style='{'))
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)

__version__ = "0.0.1"
