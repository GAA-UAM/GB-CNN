""" Generating logs """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import logging
import sys


class FileHandler:
    logger = logging.getLogger('Monitor')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('%s.log' % 'log', 'a')
    formatter = logging.Formatter(
        '%(asctime)s  - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    @classmethod
    def debug(cls, msg):
        cls.logger.debug(msg)

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)


class StreamHandler:

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info('This is a log message!')

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)
