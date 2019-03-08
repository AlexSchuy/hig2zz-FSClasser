"""Validate

This module contains validation methods that assert various common conditions.
"""
import os

import numpy as np
import pandas as pd
from config import settings
from data import process


def check_is_array(a):
    assert isinstance(a, pd.DataFrame) or isinstance(a, pd.Series) or isinstance(
        a, np.ndarray), 'invalid data set of type {}'.format(type(a))


def check_version(v):
    assert v in process.VERSIONS, 'v={}'.format(v)


def check_consistent_length(a1, a2):
    assert a1.shape[0] == a2.shape[0], '{} != {}'.format(
        a1.shape[0], a2.shape[0])


def check_dataset(version, dataset):
    dataset_path = os.path.join(settings.get_setting('Event Selection', 'merged_fs'), version, dataset)
    assert os.path.isdir(
        dataset_path), '{} is not a valid directory!'.format(dataset_path)
