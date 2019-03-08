"""Cache

This module contains methods for managing the on-disk cache of final state data.
This includes retrieving data from FSClasser root files, calculating derived
variables, and saving data to pickled files that are more easily accessible in
python. 

Note that other modules should use the data module to load data rather than this
module.

"""

import glob
import logging
import math
import os
from collections import defaultdict

from config import settings
import numpy as np
import pandas as pd
from data import process
import progressbar
import uproot
from common import utils, validate

MERGED_INPUT_DIR = settings.get_setting('Event Selection', 'merged_fs')
CACHE_DIR = settings.get_setting('Event Selection', 'cache_dir')
MUON_PID = 13
META_KEYS = ['pid', 'rfid']
DERIVED_KEYS = ('LeadingMuonPt', 'SubLeadingMuonPt', 'LeadingMuonCosTheta', 'SubLeadingMuonCosTheta', 'LeadingMuonPhi', 'SubLeadingMuonPhi', 'LeadingJetPt', 'SubLeadingJetPt', 'LeadingJetCosTheta', 'SubLeadingJetCosTheta', 'LeadingJetPhi', 'SubLeadingJetPhi', 'LeadingJetnPFO', 'SubLeadingJetnPFO', 'JetPhiP1', 'JetPhiP2', 'PfoPhiP3', 'PfoPhiP4', 'MissingMass', 'PtScalarSum', 'EventMultiplicity', 'En1234', 'Px1234', 'Py1234', 'Pz1234', 'P1234', 'gamma', 'JetEnP12', 'JetPxP12', 'JetPyP12', 'JetPzP12', 'PfoEnP34', 'PfoPxP34', 'PfoPyP34', 'PfoPzP34', 'JetPxCOMP1', 'JetPyCOMP1', 'JetPzCOMP1', 'JetPxCOMP2', 'JetPyCOMP2', 'JetPzCOMP2', 'PfoPxCOMP3', 'PfoPyCOMP3', 'PfoPzCOMP3', 'PfoPxCOMP4', 'PfoPyCOMP4', 'PfoPzCOMP4', 'JetPxCOMP12', 'JetPyCOMP12', 'JetPzCOMP12', 'PfoPxCOMP34', 'PfoPyCOMP34', 'PfoPzCOMP34', 'MinA1', 'MinA2', 'Anlj', 'CosTheta', 'VisPt', 'NonDiMuonVisEn', 'NonDiMuonVisPx', 'NonDiMuonVisPy', 'NonDiMuonVisPz', 'NonDiMuonVisMass', 'MinA1Lab', 'MinA2Lab', 'LeadingEnMuonEn', 'SubLeadingEnMuonEn', 'LeadingEnJetEn', 'SubLeadingEnJetEn', 'InvisMass', 'mc_zmm_m', 'mc_zqq_m')

def _get_root_filepaths(version, dataset, processes):
    """Returns a list of paths to all root files for each process in processes
    for the given CEPC version and dataset."""
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    if isinstance(processes, str):
        processes = [processes]

    return [sorted(glob.glob(os.path.join(MERGED_INPUT_DIR, version, dataset, p, '*.root'))) for p in processes]

def _get_cache_filepaths(version, dataset, processes):
    """Returns the path of each pickle cache file for each process in processes
    for the given CEPC version and dataset."""

    if isinstance(processes, str):
        processes = [processes]

    utils.makedirs(os.path.join(CACHE_DIR, version, dataset))

    return [os.path.join(CACHE_DIR, version, dataset, p+'.pkl') for p in processes]

def load_data(data_keys, version, dataset, processes, use_progressbar=True):
    """Loads the data associated with the given data_keys for each process in
    processes for the given CEPC version and dataset.

    For any data_keys that are not found in the cache, it saves them to the 
    cache (see _save_cache() for details) and tries again.

    Returns (features, meta) where features is a dataframe that
    contains the requested data and meta is an ancillary dataframe that contains
    id information. Specifically, meta contains the following columns:

    'pid': int
        An int associated with a process string as determined by 
        process.get_pid_map(version, dataset).
    'rfid': int
        An int corresponding to a root file for a given pid.

    Parameters
    ----------
    data_keys: string or array_like
        The data keys to load.

    version: string
        The CEPC version to use.

    dataset: string
        The name of the data set to use.

    processes: string or array_like
        The particle decay processes.

    use_progressbar: bool, optional (default=True)
        If set, write a textual progressbar to standard output to indicate
        loading progress.
    """
    vprogressbar = (lambda i, **kwds: progressbar.progressbar(i, redirect_stdout=True, **kwds)) if use_progressbar else (lambda i, **kwds: i)
    if isinstance(data_keys, str):
        data_keys = [data_keys]
    if isinstance(processes, str):
        processes = [processes]
    logging.info('Loading {} from {} for {}.'.format(str(data_keys), processes, dataset))

    # Sort the processes to ensure consistent ordering.
    processes = sorted(processes)

    # Check for missing pickle files.
    cache_filepaths = _get_cache_filepaths(version, dataset, processes)
    missing_processes = [f for c,f in zip(cache_filepaths, processes) if not os.path.isfile(c)]
    if missing_processes:
        logging.warning('{} for {} missing from the cache.'.format(missing_processes, dataset))
        _save_cache(data_keys, version, dataset, missing_processes, use_progressbar=use_progressbar)

    # Load from the pickle files.
    data = pd.DataFrame()
    meta = pd.DataFrame()
    for cache_filepath, p in vprogressbar(zip(cache_filepaths, processes), max_value=len(processes)):

        # Read from the cache. If the cache is empty, skip to the next file.
        df = pd.read_pickle(cache_filepath)
        if df.empty:
            continue

        # Check to see if we need to fix the pid.
        if df['pid'][0] != process.get_pid_map(version, dataset)[p]:
            logging.warning('pid change detected, fixing...')
            df['pid'] = process.get_pid_map(version, dataset)[p]
            df.to_pickle(cache_filepath)

        # If there are any missing keys, save them to the cache and try again.
        missing_keys = [dk for dk in data_keys if dk not in df]
        if missing_keys:
            logging.debug('{} missing {}'.format(str(p), str(missing_keys)))
            _save_cache(missing_keys, version, dataset, p, use_progressbar=False)
            df = pd.read_pickle(cache_filepath)
        data = pd.concat([data, df[data_keys]], ignore_index=True)
        meta = pd.concat([meta, df[META_KEYS]], ignore_index=True)


    # Ensure that data is returned in a consistent order.
    data.sort_index(axis=1, inplace=True)
    meta.sort_index(axis=1, inplace=True)

    return data, meta

def _save_cache(data_keys, version, dataset, processes, use_progressbar=True):
    """Update the cache of each file with the given data keys.

    For any underived keys, collect results from the appropriate root files. For
    any derived keys, call their appropriate generation function and recursively 
    call load_data to obtain necessary features. This may result in _save_cache
    being recursively called, thus, check the cache before saving data.

    This function also saves meta information to the cache, which is as described
    in load_data().

    Parameters are as described in load_data().
    """
    vprogressbar = (lambda i, **kwds: progressbar.progressbar(i, redirect_stdout=True, **kwds)) if use_progressbar else (lambda i, **kwds: i)
    if isinstance(data_keys, str):
        data_keys = [data_keys]
    if isinstance(processes, str):
        processes = [processes]

    derived_keys = [dk for dk in data_keys if dk in DERIVED_KEYS]
    data_keys = [dk for dk in data_keys if dk not in DERIVED_KEYS]
    # If any file is missing rfid, load it.
    for cache_filepath in _get_cache_filepaths(version, dataset, processes):
        if not os.path.isfile(cache_filepath):
            data_keys.append('rfid')
            break
    if data_keys:
        logging.info('Saving {} from {} to cache from root files...'.format(str(data_keys), str(processes)))
    for p, cache_filepath, root_filepaths in vprogressbar(zip(processes, _get_cache_filepaths(version, dataset, processes), _get_root_filepaths(version, dataset, processes)), max_value=len(processes)):
        data = pd.DataFrame()
        dicts = []
        # Collect un-derived data from the appropriate root files.
        if data_keys:
            for f in root_filepaths:
                try:
                    dicts.append(uproot.open(f)['ntINC2_0001100'].arrays(branches=data_keys))
                except KeyError:
                    logging.warning('({}, {}) has an empty root file.'.format(version, p))
                except ValueError as e:
                    logging.error('Unable to read {} due to: {}'.format(f, e))

            # If no root file could be successfully read, assume that there are no
            # entries for this process.
            if not dicts:
                logging.warning('({}, {}) has no entries.'.format(version, p))
                data.to_pickle(cache_filepath)
                continue

            arrays = {k: np.concatenate([d[k.encode('UTF-8')] for d in dicts]) for k in data_keys}
            data = pd.DataFrame.from_dict(arrays)

        # Calculate derived keys.
        if derived_keys:
            logging.info('Calculating {} for {}.'.format(str(derived_keys), str(processes)))
            derived = _generate_derived_data(derived_keys, version, dataset, p)
            data = pd.concat([data, derived], axis=1)

        # Eliminate any keys that were required for derived data (and thus saved to
        # the cache).
        if os.path.isfile(cache_filepath):
            cached = pd.read_pickle(cache_filepath)
            data = pd.concat([data, cached[cached.columns.difference(data.columns)]], axis=1)

        # Calculate meta information if necessary.
        if 'pid' not in data:
            data['pid'] = process.get_pid_map(version, dataset)[p]

        data.sort_index(axis=1, inplace=True)
        data.to_pickle(cache_filepath)

def _generate_derived_data(derived_keys, version, dataset, process):
    data = pd.DataFrame()
    for derived_key in derived_keys:
        data[derived_key] = eval('_generate_{}'.format(derived_key))(version, dataset, process)
    return data

def _leading_en_muon_mask(version, dataset, process):
    muon_ens, _ = load_data(['PfoEnP3', 'PfoEnP4'], version, dataset, process, use_progressbar=False)
    return (muon_ens['PfoEnP4'] > muon_ens['PfoEnP3'])

def _leading_en_jet_mask(version, dataset, process):
    jet_ens, _ = load_data(['JetEnP1', 'JetEnP2'], version, dataset, process, use_progressbar=False)
    return (jet_ens['JetEnP2'] > jet_ens['JetEnP1'])

def _generate_LeadingEnMuonEn(version, dataset, process):
    return _select_with_mask('PfoEnP3', 'PfoEnP4', _leading_en_muon_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingEnMuonEn(version, dataset, process):
    return _select_with_mask('PfoEnP4', 'PfoEnP3', _leading_en_muon_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingEnJetEn(version, dataset, process):
    return _select_with_mask('JetEnP1', 'JetEnP2', _leading_en_jet_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingEnJetEn(version, dataset, process):
    return _select_with_mask('JetEnP2', 'JetEnP1', _leading_en_jet_mask(version, dataset, process), version, dataset, process)

def _leading_muon_mask(version, dataset, process):
    muon_pts, _ = load_data(['PfoPtP3', 'PfoPtP4'], version, dataset, process, use_progressbar=False)
    return (muon_pts['PfoPtP4'] > muon_pts['PfoPtP3'])

def _leading_jet_mask(version, dataset, process):
    jet_pts, _ = load_data(['JetPtP1', 'JetPtP2'], version, dataset, process, use_progressbar=False)
    return (jet_pts['JetPtP2'] > jet_pts['JetPtP1'])

def _select_with_mask(key1, key2, mask, version, dataset, process):
    d, _ = load_data([key1, key2], version, dataset, process, use_progressbar=False)
    return d[key2].where(mask, d[key1])

def _generate_LeadingMuonPt(version, dataset, process):
    return _select_with_mask('PfoPtP3', 'PfoPtP4', _leading_muon_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingMuonPt(version, dataset, process):
    return _select_with_mask('PfoPtP4', 'PfoPtP3', _leading_muon_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingMuonCosTheta(version, dataset, process):
    return _select_with_mask('PfocosThetaP3', 'PfocosThetaP4', _leading_muon_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingMuonCosTheta(version, dataset, process):
    return _select_with_mask('PfocosThetaP4', 'PfocosThetaP3', _leading_muon_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingMuonPhi(version, dataset, process):
    return _select_with_mask('PfoPhiP3', 'PfoPhiP4', _leading_muon_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingMuonPhi(version, dataset, process):
    return _select_with_mask('PfoPhiP4', 'PfoPhiP3', _leading_muon_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingJetPt(version, dataset, process):
    return _select_with_mask('JetPtP1', 'JetPtP2', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingJetPt(version, dataset, process):
    return _select_with_mask('JetPtP2', 'JetPtP1', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingJetCosTheta(version, dataset, process):
    return _select_with_mask('JetcosThetaP1', 'JetcosThetaP2', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingJetCosTheta(version, dataset, process):
    return _select_with_mask('JetcosThetaP2', 'JetcosThetaP1', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingJetPhi(version, dataset, process):
    return _select_with_mask('JetPhiP1', 'JetPhiP2', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingJetPhi(version, dataset, process):
    return _select_with_mask('JetPhiP2', 'JetPhiP1', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_LeadingJetnPFO(version, dataset, process):
    return _select_with_mask('JetnPFOP1', 'JetnPFOP2', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_SubLeadingJetnPFO(version, dataset, process):
    return _select_with_mask('JetnPFOP2', 'JetnPFOP1', _leading_jet_mask(version, dataset, process), version, dataset, process)

def _generate_JetPhiP1(version, dataset, process):
    jet_p, _ = load_data(['JetPxP1', 'JetPyP1'], version, dataset, process, use_progressbar=False)
    return np.mod(np.arctan2(jet_p['JetPyP1'], jet_p['JetPxP1']), 2 * math.pi)

def _generate_JetPhiP2(version, dataset, process):
    jet_p, _ = load_data(['JetPxP2', 'JetPyP2'], version, dataset, process, use_progressbar=False)
    return np.mod(np.arctan2(jet_p['JetPyP2'], jet_p['JetPxP2']), 2 * math.pi)

def _generate_PfoPhiP3(version, dataset, process):
    muon_p, _ = load_data(['PfoPxP3', 'PfoPyP3'], version, dataset, process, use_progressbar=False)
    return np.mod(np.arctan2(muon_p['PfoPyP3'], muon_p['PfoPxP3']), 2 * math.pi)

def _generate_PfoPhiP4(version, dataset, process):
    muon_p, _ = load_data(['PfoPxP4', 'PfoPyP4'], version, dataset, process, use_progressbar=False)
    return np.mod(np.arctan2(muon_p['PfoPyP4'], muon_p['PfoPxP4']), 2 * math.pi)

def _generate_MissingMass(version, dataset, process):
    missing_mass_squared, _ = load_data(['MissingMass2'], version, dataset, process, use_progressbar=False)
    missing_mass = missing_mass_squared['MissingMass2']**(1/2)
    null_mask = pd.isnull(missing_mass)
    missing_mass[null_mask] = -(-missing_mass_squared['MissingMass2'][null_mask])**(1/2)
    return missing_mass

def _generate_PtScalarSum(version, dataset, process):
    Pts, _ = load_data(['JetPtP1', 'JetPtP2', 'PfoPtP3', 'PfoPtP4'], version, dataset, process, use_progressbar=False)
    return Pts['JetPtP1'] + Pts['JetPtP2'] + Pts['PfoPtP3'] + Pts['PfoPtP4']

def _generate_EventMultiplicity(version, dataset, process):
    event, meta = load_data('Event', version, dataset, process, use_progressbar=False)
    event_counts = defaultdict(int)
    event_multiplicity = np.zeros(event.shape[0])
    for i in range(event.shape[0]):
        pid = meta.loc[i, 'pid']
        rfid = meta.loc[i, 'rfid']
        event_num = event.loc[i, 'Event']
        event_counts[(pid, rfid, event_num)] += 1
    for i in range(event.shape[0]):
        pid = meta.loc[i, 'pid']
        rfid = meta.loc[i, 'rfid']
        event_num = event.loc[i, 'Event']
        event_multiplicity[i] = event_counts[(pid, rfid, event_num)]
    return event_multiplicity

def _generate_En1234(version, dataset, process):
    df, _ = load_data(['JetEnP1', 'JetEnP2', 'PfoEnP3', 'PfoEnP4'], version, dataset, process, use_progressbar=False)
    return df['JetEnP1'] + df['JetEnP2'] + df['PfoEnP3'] + df['PfoEnP4']

def _generate_Px1234(version, dataset, process):
    df, _ = load_data(['JetPxP1', 'JetPxP2', 'PfoPxP3', 'PfoPxP4'], version, dataset, process, use_progressbar=False)
    return df['JetPxP1'] + df['JetPxP2'] + df['PfoPxP3'] + df['PfoPxP4']

def _generate_Py1234(version, dataset, process):
    df, _ = load_data(['JetPyP1', 'JetPyP2', 'PfoPyP3', 'PfoPyP4'], version, dataset, process, use_progressbar=False)
    return df['JetPyP1'] + df['JetPyP2'] + df['PfoPyP3'] + df['PfoPyP4']

def _generate_Pz1234(version, dataset, process):
    df, _ = load_data(['JetPzP1', 'JetPzP2', 'PfoPzP3', 'PfoPzP4'], version, dataset, process, use_progressbar=False)
    return df['JetPzP1'] + df['JetPzP2'] + df['PfoPzP3'] + df['PfoPzP4']

def _generate_P1234(version, dataset, process):
    df, _ = load_data(['Px1234', 'Py1234', 'Pz1234'], version, dataset, process, use_progressbar=False)
    return (df['Px1234']**2 + df['Py1234']**2 + df['Pz1234']**2)**(1/2)

def _generate_gamma(version, dataset, process):
    df, _ = load_data(['En1234', 'RMass1234'], version, dataset, process, use_progressbar=False)
    return df['En1234'] / df['RMass1234']

def _generate_JetEnP12(version, dataset, process):
    df, _ = load_data(['JetEnP1', 'JetEnP2'], version, dataset, process, use_progressbar=False)
    return df['JetEnP1'] + df['JetEnP2']

def _generate_JetPxP12(version, dataset, process):
    df, _ = load_data(['JetPxP1', 'JetPxP2'], version, dataset, process, use_progressbar=False)
    return df['JetPxP1'] + df['JetPxP2']

def _generate_JetPyP12(version, dataset, process):
    df, _ = load_data(['JetPyP1', 'JetPyP2'], version, dataset, process, use_progressbar=False)
    return df['JetPyP1'] + df['JetPyP2']

def _generate_JetPzP12(version, dataset, process):
    df, _ = load_data(['JetPzP1', 'JetPzP2'], version, dataset, process, use_progressbar=False)
    return df['JetPzP1'] + df['JetPzP2']

def _generate_PfoEnP34(version, dataset, process):
    df, _ = load_data(['PfoEnP3', 'PfoEnP4'], version, dataset, process, use_progressbar=False)
    return df['PfoEnP3'] + df['PfoEnP4']

def _generate_PfoPxP34(version, dataset, process):
    df, _ = load_data(['PfoPxP3', 'PfoPxP4'], version, dataset, process, use_progressbar=False)
    return df['PfoPxP3'] + df['PfoPxP4']

def _generate_PfoPyP34(version, dataset, process):
    df, _ = load_data(['PfoPyP3', 'PfoPyP4'], version, dataset, process, use_progressbar=False)
    return df['PfoPyP3'] + df['PfoPyP4']

def _generate_PfoPzP34(version, dataset, process):
    df, _ = load_data(['PfoPzP3', 'PfoPzP4'], version, dataset, process, use_progressbar=False)
    return df['PfoPzP3'] + df['PfoPzP4']

def _Pn_COM_boosted(i, n, version, dataset, process):
    '''Returns the ith component of the four-momenta for the nth particle.'''
    if n == 1 or n == 2 or n == 12:
        n = str(n)
        E_name = 'JetEnP'+n
        px_name = 'JetPxP'+n
        py_name = 'JetPyP'+n
        pz_name = 'JetPzP'+n
    elif n == 3 or n == 4 or n == 34:
        n = str(n)
        E_name = 'PfoEnP'+n
        px_name = 'PfoPxP'+n
        py_name = 'PfoPyP'+n
        pz_name = 'PfoPzP'+n
    df, _ = load_data(['gamma', 'En1234', 'Px1234', 'Py1234', 'Pz1234', 'RMass1234', 'P1234', E_name, px_name, py_name, pz_name], version, dataset, process, use_progressbar=False)
    gamma = df['gamma']
    E = ['En1234']
    M = df['RMass1234']
    px = df['Px1234']
    py = df['Py1234']
    pz = df['Pz1234']
    p = df['P1234']
    E1 = df[E_name]
    px1 = df[px_name]
    py1 = df[py_name]
    pz1 = df[pz_name]
    if i == 0:
        return gamma*E1 - px/M*px1 - py/M*py1 - pz/M*pz1
    if i == 1:
        return (-px/M)*E1 + (1+(gamma-1)*px**2/p**2)*px1 + ((gamma-1)*px*py/p**2)*py1 + ((gamma-1)*px*pz/p**2)*pz1
    if i == 2:
        return (-py/M)*E1 + ((gamma-1)*py*px/p**2)*px1 + (1+(gamma-1)*py**2/p**2)*py1 + ((gamma-1)*py*pz/p**2)*pz1
    if i == 3:
        return (-pz/M)*E1 + ((gamma-1)*pz*px/p**2)*px1 + ((gamma-1)*pz*py/p**2)*py1 + (1+(gamma-1)*pz**2/p**2)*pz1

def _generate_JetPxCOMP1(version, dataset, process):
    return _Pn_COM_boosted(1, 1, version, dataset, process)

def _generate_JetPyCOMP1(version, dataset, process):
    return _Pn_COM_boosted(2, 1, version, dataset, process)

def _generate_JetPzCOMP1(version, dataset, process):
    return _Pn_COM_boosted(3, 1, version, dataset, process)

def _generate_JetPxCOMP2(version, dataset, process):
    return _Pn_COM_boosted(1, 2, version, dataset, process)

def _generate_JetPyCOMP2(version, dataset, process):
    return _Pn_COM_boosted(2, 2, version, dataset, process)

def _generate_JetPzCOMP2(version, dataset, process):
    return _Pn_COM_boosted(3, 2, version, dataset, process)

def _generate_PfoPxCOMP3(version, dataset, process):
    return _Pn_COM_boosted(1, 3, version, dataset, process)

def _generate_PfoPyCOMP3(version, dataset, process):
    return _Pn_COM_boosted(2, 3, version, dataset, process)

def _generate_PfoPzCOMP3(version, dataset, process):
    return _Pn_COM_boosted(3, 3, version, dataset, process)

def _generate_PfoPxCOMP4(version, dataset, process):
    return _Pn_COM_boosted(1, 4, version, dataset, process)

def _generate_PfoPyCOMP4(version, dataset, process):
    return _Pn_COM_boosted(2, 4, version, dataset, process)

def _generate_PfoPzCOMP4(version, dataset, process):
    return _Pn_COM_boosted(3, 4, version, dataset, process)

def _generate_JetPxCOMP12(version, dataset, process):
    return _Pn_COM_boosted(1, 12, version, dataset, process)

def _generate_JetPyCOMP12(version, dataset, process):
    return _Pn_COM_boosted(2, 12, version, dataset, process)

def _generate_JetPzCOMP12(version, dataset, process):
    return _Pn_COM_boosted(3, 12, version, dataset, process)

def _generate_PfoPxCOMP34(version, dataset, process):
    return _Pn_COM_boosted(1, 34, version, dataset, process)

def _generate_PfoPyCOMP34(version, dataset, process):
    return _Pn_COM_boosted(2, 34, version, dataset, process)

def _generate_PfoPzCOMP34(version, dataset, process):
    return _Pn_COM_boosted(3, 34, version, dataset, process)

def _angle(x1, y1, z1, x2, y2, z2):
    return np.arccos(np.clip((x1*x2+y1*y2+z1*z2)/((x1**2+y1**2+z1**2)**(1/2)*(x2**2+y2**2+z2**2)**(1/2)), -1, 1))

def _MinAnLab(n, version, dataset, process):
    df, _ = load_data(['JetPxP1', 'JetPyP1', 'JetPzP1', 'JetPxP2', 'JetPyP2', 'JetPzP2', 'PfoPxP3', 'PfoPyP3', 'PfoPzP3', 'PfoPxP4', 'PfoPyP4', 'PfoPzP4'], version, dataset, process, use_progressbar=False)
    px1 = df['JetPxP1']
    py1 = df['JetPyP1']
    pz1 = df['JetPzP1']
    px2 = df['JetPxP2']
    py2 = df['JetPyP2']
    pz2 = df['JetPzP2']
    px3 = df['PfoPxP3']
    py3 = df['PfoPyP3']
    pz3 = df['PfoPzP3']
    px4 = df['PfoPxP4']
    py4 = df['PfoPyP4']
    pz4 = df['PfoPzP4']
    a1 = _angle(px1, py1, pz1, px3, py3, pz3)
    a2 = _angle(px1, py1, pz1, px4, py4, pz4)
    a3 = _angle(px2, py2, pz2, px3, py3, pz3)
    a4 = _angle(px2, py2, pz2, px4, py4, pz4)
    if n == 1:
        return np.minimum.reduce([a1, a2, a3, a4])
    if n == 2:
        _angles = np.array([a1, a2, a3, a4])
        argmin = np.argmin(_angles, axis=0)
        return _angles[3 - argmin, np.arange(_angles.shape[1])]

def _generate_MinA1Lab(version, dataset, process):
    return _MinAnLab(1, version, dataset, process)

def _generate_MinA2Lab(version, dataset, process):
    return _MinAnLab(2, version, dataset, process)

def _MinAn(n, version, dataset, process):
    df, _ = load_data(['JetPxCOMP1', 'JetPyCOMP1', 'JetPzCOMP1', 'JetPxCOMP2', 'JetPyCOMP2', 'JetPzCOMP2', 'PfoPxCOMP3', 'PfoPyCOMP3', 'PfoPzCOMP3', 'PfoPxCOMP4', 'PfoPyCOMP4', 'PfoPzCOMP4'], version, dataset, process, use_progressbar=False)
    px1 = df['JetPxCOMP1']
    py1 = df['JetPyCOMP1']
    pz1 = df['JetPzCOMP1']
    px2 = df['JetPxCOMP2']
    py2 = df['JetPyCOMP2']
    pz2 = df['JetPzCOMP2']
    px3 = df['PfoPxCOMP3']
    py3 = df['PfoPyCOMP3']
    pz3 = df['PfoPzCOMP3']
    px4 = df['PfoPxCOMP4']
    py4 = df['PfoPyCOMP4']
    pz4 = df['PfoPzCOMP4']
    a1 = _angle(px1, py1, pz1, px3, py3, pz3)
    a2 = _angle(px1, py1, pz1, px4, py4, pz4)
    a3 = _angle(px2, py2, pz2, px3, py3, pz3)
    a4 = _angle(px2, py2, pz2, px4, py4, pz4)
    if n == 1:
        return np.minimum.reduce([a1, a2, a3, a4])
    if n == 2:
        _angles = np.array([a1, a2, a3, a4])
        argmin = np.argmin(_angles, axis=0)
        return _angles[3 - argmin, np.arange(_angles.shape[1])]

def _generate_MinA1(version, dataset, process):
    return _MinAn(1, version, dataset, process)

def _generate_MinA2(version, dataset, process):
    return _MinAn(2, version, dataset, process)

def _generate_Anlj(version, dataset, process):
    df, _ = load_data(['JetPxP12', 'JetPyP12', 'JetPzP12', 'PfoPxP34', 'PfoPyP34', 'PfoPzP34'], version, dataset, process, use_progressbar=False)
    return _angle(df['JetPxP12'], df['JetPyP12'], df['JetPzP12'], df['PfoPxP34'], df['PfoPyP34'], df['PfoPzP34'])

def _generate_CosTheta(version, dataset, process):
    df, _ = load_data(['TotalPy', 'TotalPz'], version, dataset, process, use_progressbar=False)
    return df['TotalPz'] / (df['TotalPy']**2 + df['TotalPz']**2)**(1/2)

def _generate_VisPt(version, dataset, process):
    df, _ = load_data(['VisPx', 'VisPy'], version, dataset, process, use_progressbar=False)
    return (df['VisPx']**2 + df['VisPy']**2)**(1/2)

def _generate_NonDiMuonVisEn(version, dataset, process):
    df, _ = load_data(['VisEn', 'PfoEnP34'], version, dataset, process, use_progressbar=False)
    return df['VisEn'] - df['PfoEnP34']

def _generate_NonDiMuonVisPx(version, dataset, process):
    df, _ = load_data(['VisPx', 'PfoPxP34'], version, dataset, process, use_progressbar=False)
    return df['VisPx'] - df['PfoPxP34']

def _generate_NonDiMuonVisPy(version, dataset, process):
    df, _ = load_data(['VisPy', 'PfoPyP34'], version, dataset, process, use_progressbar=False)
    return df['VisPy'] - df['PfoPyP34']

def _generate_NonDiMuonVisPz(version, dataset, process):
    df, _ = load_data(['VisPz', 'PfoPzP34'], version, dataset, process, use_progressbar=False)
    return df['VisPz'] - df['PfoPzP34']

def _generate_NonDiMuonVisMass(version, dataset, process):
    df, _ = load_data(['NonDiMuonVisEn', 'NonDiMuonVisPx', 'NonDiMuonVisPy', 'NonDiMuonVisPz'], version, dataset, process, use_progressbar=False)
    return (df['NonDiMuonVisEn']**2 - df['NonDiMuonVisPx']**2 - df['NonDiMuonVisPy']**2 - df['NonDiMuonVisPz']**2)**(1/2)

def _generate_InvisMass(version, dataset, process):
    df, _ = load_data(['VisEn', 'VisPx', 'VisPy', 'VisPz'], version, dataset, process, use_progressbar=False)
    if version == 'v4':
        ecms = 240
    elif version == 'v1':
        ecms = 250
    invis_mass_squared = (ecms - df['VisEn'])**2 - df['VisPx']**2 - df['VisPy']**2 - df['VisPz']**2
    invis_mass = invis_mass_squared**(1/2)
    null_mask = pd.isnull(invis_mass)
    invis_mass[null_mask] = -(-invis_mass_squared[null_mask])**(1/2)
    return invis_mass

def _generate_VisP(version, dataset, process):
    df, _ = load_data(['VisPx', 'VisPy', 'VisPz'], version, dataset, process, use_progressbar=False)
    return (df['VisPx']**2 + df['VisPy']**2 + df['VisPz']**2)**(1/2)

def _generate_mc_zmm_m(version, dataset, process):
    df, _ = load_data(['mc_z1_m', 'mc_z2_m', 'mc_zz_flag', 'mc_z1_daughter_pid', 'mc_z2_daughter_pid'], version, dataset, process, use_progressbar=False)
    mc_zmm_m = pd.Series(np.full((df.shape[0],), -1.0))
    z1_mask = np.logical_and(df['mc_zz_flag'] == 13, df['mc_z1_daughter_pid'] == MUON_PID)
    mc_zmm_m[z1_mask] = df.loc[z1_mask, 'mc_z1_m']
    z2_mask = np.logical_and(df['mc_zz_flag'] == 31, df['mc_z2_daughter_pid'] == MUON_PID)
    mc_zmm_m[z2_mask] = df.loc[z2_mask, 'mc_z2_m']
    return mc_zmm_m

def _generate_mc_zqq_m(version, dataset, process):
    df, _ = load_data(['mc_z1_m', 'mc_z2_m', 'mc_zz_flag', 'mc_z1_daughter_pid', 'mc_z2_daughter_pid'], version, dataset, process, use_progressbar=False)
    mc_zqq_m = pd.Series(np.full((df.shape[0],), -1.0))
    z1_mask = np.logical_and(df['mc_zz_flag'] == 31, df['mc_z2_daughter_pid'] == MUON_PID)
    mc_zqq_m[z1_mask] = df.loc[z1_mask, 'mc_z1_m']
    z2_mask = np.logical_and(df['mc_zz_flag'] == 13, df['mc_z1_daughter_pid'] == MUON_PID)
    mc_zqq_m[z2_mask] = df.loc[z2_mask, 'mc_z2_m']
    return mc_zqq_m


def delete_from_cache(version, dataset, processes, data_keys=None):
    if isinstance(data_keys, str):
        data_keys = [data_keys]
    if isinstance(processes, str):
        processes = [processes]
    cache_filepaths = _get_cache_filepaths(version, dataset, processes)
    for cache_filepath in cache_filepaths:
        if data_keys is None:
            try:
                os.remove(cache_filepath)
            except:
                continue
        else:
            try:
                df = pd.read_pickle(cache_filepath)
                df = df.drop(columns=data_keys)
                df.to_pickle(cache_filepath)
            except:
                continue


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Update the cache.')
    parser.add_argument('--version', '-v', default='v4', choices=process.VERSIONS, help='CEPC version to use.')
    parser.add_argument('--datasets', nargs='+', help='The data sets to operate on. If unspecified, operate on all data sets.')
    parser.add_argument('--processes', '-p', nargs='+', help='The processes within each data set to operate on. If unspecified, operate on all processes.')
    parser.add_argument('--data_keys', '-k', nargs='+', help='The data keys to operate on. If unspecified and the operation is delete, delete all saved data.')
    parser.add_argument('operation', choices=['load', 'delete'], help='The type of operation to perform.')

    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = [os.path.split(dataset)[1] for dataset in glob.glob(os.path.join(MERGED_INPUT_DIR, args.version, '*')) if os.path.isdir(dataset)]
    if args.operation == 'load':
        if args.data_keys is None:
            raise ValueError('Data keys must be specified when loading.')

    for dataset in args.datasets:
        if args.processes is None:
            processes = process.get_all_processes(args.version, dataset)
        else:
            processes = args.processes
        if args.operation == 'load':
            load_data(args.data_keys, args.version, dataset, processes)
        elif args.operation == 'delete':
            delete_from_cache(args.version, dataset, processes, args.data_keys)

if __name__ == '__main__':
    main()
