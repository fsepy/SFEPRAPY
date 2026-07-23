import logging

logger = logging.getLogger('sfeprapy')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(
        logging.Formatter(fmt='{asctime} {levelname:8.8s} [{filename:15.15s}:{lineno:05d}] {message:s}',
                          style='{'))
    logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)

from ._version import __version__
