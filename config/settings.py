import os
import sys
from configparser import ConfigParser, ExtendedInterpolation


def base_dir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_setting(section, setting, func=str):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(os.path.join(base_dir(), 'config', 'settings.ini'))
    config.set('Common', 'base', base_dir())
    return func(config.get(section, setting))
