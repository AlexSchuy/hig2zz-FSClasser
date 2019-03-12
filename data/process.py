"""Process

This module contains utility methods relating to processes. This includes
retrieving process names that correspond to various groups, separating
data based on groups, retrieving cross-sections, etc.

"""

import glob
import logging
import os
from collections import OrderedDict, namedtuple
from configparser import ConfigParser

import numpy as np

from common import utils, validate
from config import settings

MERGED_INPUT_DIR = settings.get_setting('Event Selection', 'merged_fs')
ID = namedtuple('ID', 'version process')
VERSIONS = ('v1', 'v4')
WEIGHT_TYPES = ('normalized', 'expected', 'unweighted')
GROUP_TYPES = ('ungrouped', 'all_background', 'sl_separated', 'standard')


def get_cross_section(version, process):
    """Returns the cross-section for the given process in the given CEPC version
    according to:
    http://cepcsoft.ihep.ac.cn/guides/Generation/docs/ExistingSamples/
    """
    validate.check_version(version)
    config = ConfigParser()
    try:
        config.read('config/cross_sections.ini')
    except Exception:
        logging.error('Unable to find cross_sections.ini file!')
        raise
    cross_section = float(config[version][process])
    return cross_section


def get_num_expected_events(version, dataset, process, int_L=5600.0):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return int_L * get_cross_section(version, process)


def get_num_MC_events(version, dataset, process, test_fraction=1.0):
    """Get the total number of Monte Carlo events that have been analyzed for 
    the given process for the given version and dataset.
    """
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    config = ConfigParser()
    try:
        config.read(os.path.join(MERGED_INPUT_DIR,
                                 version, 'num_events.ini'))
    except Exception:
        logging.error('Unable to find num_events.ini!')
        raise
    return config.getint(dataset, process) * test_fraction


def calculate_weight(version, dataset, process, int_L=5600.0, test_fraction=1.0):
    """Return a weight to scale to expected yield."""
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    N_expected = get_num_expected_events(version, dataset, process, int_L)
    N_actual = get_num_MC_events(version, dataset, process, test_fraction)
    logging.debug('{} {}: weight = {:.1f} / {:.1f} = {:.1f}'.format(version,
                                                                    process, N_expected, N_actual, N_expected / N_actual))
    return N_expected / N_actual


def get_pid_map(version, dataset):
    validate.check_dataset(version, dataset)
    processes = get_all_processes(version, dataset)
    id_map = {}
    for i, p in enumerate(processes):
        id_map[p] = i
    for k, v in [(k, v) for k, v in id_map.items()]:
        id_map[v] = k
    return id_map


def get_all_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return sorted([os.path.basename(d) for d in glob.glob(os.path.join(MERGED_INPUT_DIR, version, dataset, '*')) if os.path.isdir(d)])


def get_processes(prefixes, version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    all_processes = get_all_processes(version, dataset)
    return [g for g in all_processes if g.startswith(prefixes)]


def get_signal_process(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)

    if version == 'v1':
        raise ValueError
    else:
        if dataset == 'cross_check':
            return 'llh_zz'
        else:
            return 'nnh_zz'


def get_background_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    signal_process = get_signal_process(version, dataset)
    all_processes = get_all_processes(version, dataset)
    return [p for p in all_processes if p != signal_process]


def get_zz_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('zz_h', 'zz_l', 'zz_sl')
    return get_processes(prefixes, version, dataset)


def get_ww_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('ww_h', 'ww_l', 'ww_sl')
    return get_processes(prefixes, version, dataset)


def get_sze_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('sze_l', 'sze_sl')
    return get_processes(prefixes, version, dataset)


def get_sznu_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('sznu_l', 'sznu_sl')
    return get_processes(prefixes, version, dataset)


def get_sw_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('sw_l', 'sw_sl')
    return get_processes(prefixes, version, dataset)


def get_zzorww_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('zzorww_l', 'zzorww_h')
    return get_processes(prefixes, version, dataset)


def get_szeorsw_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('szeorsw_l',)
    return get_processes(prefixes, version, dataset)


def get_sm_leptonic_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('zz_l', 'ww_l', 'sze_l', 'sznu_l',
                'sw_l', 'szeorsw_l', 'zzorww_l')
    return get_processes(prefixes, version, dataset)


def get_sm_hadronic_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    prefixes = ('zz_h', 'ww_h', 'zzorww_h')
    return get_processes(prefixes, version, dataset)


def get_sm_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return [x for l in (get_zz_processes(version, dataset), get_ww_processes(version, dataset), get_sze_processes(version, dataset), get_sznu_processes(version, dataset), get_sw_processes(version, dataset)) for x in l]


def get_zh_processes(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    signal = get_signal_process(version, dataset)
    prefixes = tuple(p for p in ['llh_zz', 'nnh_zz'] if p != signal)
    return get_processes(prefixes, version, dataset)


def get_groups(version, dataset, group_type):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    assert group_type in GROUP_TYPES

    if group_type == 'ungrouped':
        return OrderedDict([(k, [k]) for k in get_all_processes(version, dataset)])
    if group_type == 'all_background':
        return get_signal_background_groups(version, dataset)
    if group_type == 'sl_separated':
        return get_sl_separated_groups(version, dataset)
    if group_type == 'standard':
        return get_standard_groups(version, dataset)


def get_standard_groups(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return OrderedDict([(get_signal_process(version, dataset), [get_signal_process(version, dataset)]), ('zz', get_zz_processes(version, dataset)), ('zzorww', get_zzorww_processes(version, dataset)), ('ww', get_ww_processes(version, dataset)), ('sw', get_sw_processes(version, dataset)), ('sze', get_sze_processes(version, dataset)), ('szeorsw', get_szeorsw_processes(version, dataset)), ('sznu', get_sznu_processes(version, dataset))])


def get_sl_separated_groups(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return OrderedDict([(get_signal_process(version, dataset), [get_signal_process(version, dataset)]), ('SM_leptonic', get_sm_leptonic_processes(version, dataset)), ('SM_hadronic', get_sm_hadronic_processes(version, dataset)), ('sznu_sl', get_processes(['sznu_sl'], version, dataset)), ('sze_sl', get_processes(['sze_sl'], version, dataset)), ('ww_sl', get_processes(['ww_sl'], version, dataset)), ('zz_sl', get_processes(['zz_sl'], version, dataset))])


def get_signal_background_groups(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return OrderedDict([('signal', [get_signal_process(version, dataset)]), ('background', get_background_processes(version, dataset))])


def get_zh_background_groups(version, dataset):
    signal = get_signal_process(version, dataset)
    prefixes = [p for p in [('llh_zz', 'nnh_zz')] if p != signal]
    return OrderedDict(sorted({p: get_processes(p, version, dataset) for p in prefixes}, key=lambda t: t[0]))


def get_signal_sm_zh_background_groups(version, dataset):
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    return OrderedDict([('signal', [get_signal_process(version, dataset)]), ('sm_background', get_sm_processes(version, dataset)), ('zh_background', get_zh_processes(version, dataset))])


def by_group(version, dataset, groups, pids, A=None, weight_type=None, test_fraction=1.0):
    """Separation of data sets by group.

    Let A be a collection such that A = (a1, a2, ..., an). Then, this function
    returns D = (d1, d2, ..., dn[, weights]) where d is a dictionary mapping from 
    the given groups to the corresponding entries in a. If weight_type is not
    None, weights is a similar dictionary that maps to weights calculated for each
    group according to the given weight_type.'''

    Parameters
    ----------
    version : one of VERSIONS
        CEPC version to use.

    dataset : string
        The name of the data set to use.

    groups : dict
        Maps from group names to lists of processes in the group.

    pids : array_like
        An array of process-version ids for each entry to be separated.

    A : list, optional (default=None)
        A list of data sets that should be separated. Each data set should be
        consistent in length with pids.

    weight_type : one of WEIGHT_TYPES, optional (default=None)
        The type of weights to construct. If 'expected', return weights to scale
        to expected yield (see calculate_weight()). If 'normalized', return
        weights to scale to total weight 1 in each group. If multiple processes
        are in a group, maintain relative weighting between processes in the group
        according to expected yield. If 'unweighted', return weights = 
        np.ones(pids.shape[0]).

    test_fraction : double, optional (default=1.0)
        The fraction of total samples that are used for testing. Relevant only
        if weight_type = expected.
    """

    # Validate input
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    if A is not None:
        for a in A:
            validate.check_is_array(a)
            validate.check_consistent_length(pids, a)
    assert weight_type in WEIGHT_TYPES, 'weight_type={}'.format(weight_type)

    # Separate A, if given
    D = []
    if A is not None:
        for a in A:
            d = []
            for k, v in groups.items():
                e = utils.safe_concat(
                    a[pids == get_pid_map(version, dataset)[p]] for p in v)
                if e is not None:
                    d.append((k, e))
            d = OrderedDict(d)
            D.append(d)

    # Calculate weights
    unique_ids = np.unique(pids)
    weights = np.ones(pids.shape[0])
    if weight_type != 'unweighted':
        for unique_id in unique_ids:
            weights[pids == unique_id] = calculate_weight(
                version, dataset, get_pid_map(version, dataset)[unique_id], test_fraction=test_fraction)
    weights = {k: utils.safe_concat(weights[pids == get_pid_map(version, dataset)[
                                    p]] for p in v) for k, v in groups.items()}
    empty_keys = [k for k in weights if weights[k] is None]
    for k in empty_keys:
        del weights[k]
    if weight_type == 'normalized':
        weights = {k: v / np.sum(v) for k, v in weights.items()}
    D.append(weights)

    return tuple(e for e in D)
