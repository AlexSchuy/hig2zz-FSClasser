"""Utils

This module contains miscellaneous utility methods that do not fit in another
module and may have broad use.
"""

import errno
import glob
import logging
import os
from configparser import ConfigParser

import numpy as np
import pandas as pd

from common import validate
from config import settings
from data import process


def safe_concat(arrs, default=None, **kwargs):
    """Returns the result of concatenating all non-empty elements in arrs."""
    arrs = [arr for arr in arrs]
    if not arrs:
        return default
    if isinstance(arrs[0], pd.Series):
        arrs = [arr.values for arr in arrs]
    if isinstance(arrs[0], pd.DataFrame):
        if all([arr.empty for arr in arrs]):
            return default
        return pd.concat([arr for arr in arrs if not arr.empty], **kwargs)
    if isinstance(arrs[0], np.ndarray):
        if all([arr.shape[0] == 0 for arr in arrs]):
            return default
        return np.concatenate([arr for arr in arrs if not arr.shape[0] == 0], **kwargs)


def get_binary_labels(pids, version, dataset):
    """Return a binarized version of pids, where any entry with a pid
    corresponding to the signal process is 1, and any other entry is 0.
    """
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    signal_id = process.get_pid_map(version, dataset)[
        process.get_signal_process(version, dataset)]
    labels = np.zeros_like(pids, dtype=np.float)
    labels[pids == signal_id] = 1
    return labels


def makedirs(dirs):
    try:
        os.makedirs(dirs)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_run_config(run_dir):
    config = ConfigParser(allow_no_value=True)
    config.read(os.path.join(run_dir, 'run.ini'))
    return config


def save_run_config(config, run_dir):
    with open(os.path.join(run_dir, 'run.ini'), 'w') as config_file:
        config.write(config_file)


def list_to_str(l):
    s = ''
    for item in l[:-1]:
        s += str(item) + ','
    s += l[-1]
    return s


def str_to_list(s, type_func=str):
    return [type_func(i) for i in s.split(',')]


def make_run_dir():
    RUN_DIR = settings.get_setting('Event Selection', 'run_dir')
    makedirs(RUN_DIR)
    run_number = most_recent_run_number() + 1
    run_dir = os.path.join(RUN_DIR, format(run_number, '03d'))
    try:
        os.makedirs(run_dir)
    except OSError as e:
        print(e)
    config = ConfigParser()
    with open(os.path.join(run_dir, 'run.ini'), 'w') as config_file:
        config.write(config_file)
    return run_dir


def most_recent_run_number():
    RUN_DIR = settings.get_setting('Event Selection', 'run_dir')
    return max((int(d) for d in os.listdir(RUN_DIR) if os.path.isdir(os.path.join(RUN_DIR, d))), default=-1)


def most_recent_dir():
    RUN_DIR = settings.get_setting('Event Selection', 'run_dir')
    return max((os.path.join(RUN_DIR, d) for d in os.listdir(RUN_DIR) if os.path.isdir(os.path.join(RUN_DIR, d))), key=lambda f: int(os.path.basename(f)), default=None)


def get_samples_dict(version):
    config = ConfigParser()
    config.read(settings.get_setting('Final State Selection', 'sample_table'))
    return dict(config.items(version))


def verify_samples():
    for version in ('v1', 'v4'):
        print(version+':')
        samples = get_samples_dict(version)
        for key in samples:
            if samples[key]:
                print('\t{0}: {1} files'.format(
                    key, len(glob.glob(samples[key]))))
                print('\t(path = {0}'.format(samples[key]))


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    verify_samples()
