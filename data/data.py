"""Data

This module contains methods of loading and selecting data that are used by most
other modules in the project.

"""

import logging
import os
import random
from glob import glob

from config import settings
from data import cache, process
import numpy as np
import pandas as pd
from common import utils, validate
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split

COMBINATION_TYPES = ('highest_pt', 'highest_E', 'ryuta')
SELECTION_TYPES = ('standard', 'ryuta')


def update_run_config(version, dataset, processes, feature_keys, combination_type, selection_type, selection_stage, model, run_dir):
    """Save training information in run.ini in run_dir for later use."""
    train_test_seed = random.randint(0, 2**32 - 1)
    config = utils.load_run_config(run_dir)
    config['DATA'] = {'version': version, 'dataset': dataset, 'processes': utils.list_to_str(processes), 'feature_keys': utils.list_to_str(feature_keys), 'combination_type': combination_type, 'selection_type': str(selection_type), 'selection_stage': str(selection_stage), 'model': model, 'train_test_seed': str(train_test_seed)}
    utils.save_run_config(config, run_dir)


def get_selection(pids, version, dataset, combination_type='highest_pt', selection_type='standard', use_progressbar=True):
    """Get selection information for data corresponding to the given parameters.

    For the data corresponding to the given pids, combination_type and dataset,
    return a cut dataframe, where each column corresponds to particular cut
    according to the given selection_type.

    Parameters
    ----------
    pids: array_like
        The pids that should be used (see cache.py for more information).

    version: see load_data()

    dataset: see load_data()

    combination_type: see load_data()

    selection_type: one of SELECTION_TYPES, optional (default='standard')
        The type of selection to perform. If 'standard', make selections based
        on Yuqian's prior analysis. If 'ryuta', make selections as per Ryuta's
        llh_zz analysis.

    use_progressbar: see load_data()
    """
    ids = np.unique(pids)
    processes = [process.get_pid_map(version, dataset)[i] for i in ids]
    cuts = pd.DataFrame()
    if selection_type == 'ryuta':
        selection_features, _ = load_data(['InvisMass','MissingMass', 'RMass12', 'RMass34', 'Rreco34', 'nPFOs', 'VisPt', 'MinA1Lab'], version, dataset, processes, combination_type='ryuta', use_progressbar=use_progressbar)
        # Cut 1: M(invisible) > M(jj)
        cuts['PassCut1'] = selection_features['InvisMass'] > selection_features['RMass12']
        # Cut 2: 80 GeV < M(mm) < 100 GeV
        cuts['PassCut2'] = (selection_features['RMass34'] > 80) & (selection_features['RMass34'] < 100)
        # Cut 3: 120 GeV < M_reco(mm) < 160 GeV
        cuts['PassCut3'] = (selection_features['Rreco34'] > 120) & (selection_features['Rreco34'] < 160)
        # Cut 4: nPFOs > 15
        cuts['PassCut4'] = selection_features['nPFOs'] > 15
        # Cut 5: pT(visible) > 10 GeV
        cuts['PassCut5'] = selection_features['VisPt'] > 10
        # Cut 6: MinA1Lab > 0.3 rad
        cuts['PassCut6'] = selection_features['MinA1Lab'] > 0.3
        # Cut 7: M(invisible) > 60 GeV and M(jj) < 45 GeV
        cuts['PassCut7'] = (selection_features['InvisMass'] > 60) & (selection_features['RMass12'] < 45)
    elif selection_type == 'standard':
        selection_features, _ = load_data(['nPFOs', 'VisMass', 'CosTheta'], version, dataset, processes, combination_type=combination_type, use_progressbar=use_progressbar)
        # Cut 0: nPFOs >= 10
        cuts['PassCut0'] = (selection_features['nPFOs'] >= 10)
        # Cut 1: The visible mass should be in should be in [115, 130] GeV
        cuts['PassCut1'] = (selection_features['VisMass'] >= 115) & (selection_features['VisMass'] <= 130)
        # Cut 2: |Cos(theta)| < 0.9
        cuts['PassCut2'] = (selection_features['CosTheta'].abs() < 0.9)

    return cuts

def apply_selection(A, cuts, stage=-1):
    """Mask each array in A according to the given cuts."""

    # Validate input.
    for a in A:
        validate.check_is_array(a)
        validate.check_consistent_length(a, cuts)

    # stage == -1 indicates that all stages should be used.
    if stage == -1:
        stage = cuts.columns.shape[0]

    # Use only the cuts up to the given stage.
    cuts = cuts.loc[:, cuts.columns[:stage]]

    mask = np.full(cuts.shape[0], True)
    for i, cut in enumerate(cuts):
        mask = np.logical_and(mask, cuts[cut])
        logging.info('Cut {} eliminates {}%'.format(i, 100 * (1 - np.sum(cuts[cut] / cuts.shape[0]))))

    return (a[mask] for a in A)

def select_highest_pt_combinations(features, meta, version, dataset, use_progressbar=True):
    """Select the combination from each event that involves all of the highest pT particles."""

    validate.check_consistent_length(features, meta)

    # Extract process information.
    ids = np.unique(meta['pid'])
    processes = [process.get_pid_map(version, dataset)[i] for i in ids]

    # Sort by fs pTs
    pts, _ = cache.load_data(['LeadingJetPt', 'SubLeadingJetPt', 'LeadingMuonPt', 'SubLeadingMuonPt', 'Event'], version, dataset, processes, use_progressbar=use_progressbar)
    data = pd.concat([pts[pts.columns.difference(features.columns)], features, meta], axis=1)
    data = data.sort_values(['pid', 'rfid', 'Event', 'LeadingJetPt', 'SubLeadingJetPt', 'LeadingMuonPt', 'SubLeadingMuonPt'], ascending=False)
    mask = ~data.duplicated(subset=['pid', 'rfid', 'Event'])

    return features[mask], meta[mask]

def select_highest_E_combinations(features, meta, version, dataset, use_progressbar=True):
    """Select the combination from each event that involves all of the highest energy particles."""

    validate.check_consistent_length(features, meta)

    # Extract process information.
    ids = np.unique(meta['pid'])
    processes = [process.get_pid_map(version, dataset)[i] for i in ids]

    # Sort by fs E
    E, _ = cache.load_data(['LeadingEnJetEn', 'SubLeadingEnJetEn', 'LeadingEnMuonEn', 'SubLeadingEnMuonEn', 'Event'], version, dataset, processes, use_progressbar=use_progressbar)
    data = pd.concat([E[E.columns.difference(features.columns)], features, meta], axis=1)
    data = data.sort_values(['pid', 'rfid', 'Event', 'LeadingEnJetEn', 'SubLeadingEnJetEn', 'LeadingEnMuonEn', 'SubLeadingEnMuonEn'],  ascending=False)
    mask = ~data.duplicated(subset=['pid', 'rfid', 'Event'])
    
    return features[mask], meta[mask]

def select_ryuta_combinations(features, meta, version, dataset='cross_check', use_progressbar=True):
    """For cross-checking purposes, select the combination from each event that satisfies Ryuta's preselection."""

    # Extract process information
    ids = np.unique(meta['pid'])
    processes = [process.get_pid_map(version, dataset)[i] for i in ids]

    selection_features, selection_meta = cache.load_data(['PfoEnP3', 'PfoEnP4', 'njets', 'RMass34', 'Event'], version, dataset, processes, use_progressbar=use_progressbar)

    # Select events with N(jet) = 2 and 10 GeV < E(m) < 100 GeV for m = m+, m-
    mask = (selection_features['njets'] == 2) & (selection_features['PfoEnP3'] > 10) & (selection_features['PfoEnP3'] < 100) & (selection_features['PfoEnP4'] > 10) & (selection_features['PfoEnP4'] < 100)
    selection_features = selection_features[mask]
    selection_meta = selection_meta[mask]
    features = features[mask]
    meta = meta[mask]

    # For each event, select the combination with the di-muon pair mass closest to 91.2 GeV
    selection = pd.concat([selection_features, selection_meta], axis=1)
    data = pd.concat([features, meta], axis=1)
    data = pd.concat([data, selection[selection.columns.difference(data.columns)]], axis=1)
    data['diff'] = np.absolute(data['RMass34'] - 91.2)
    data = data.sort_values(['pid', 'rfid', 'Event', 'diff'])
    mask = ~data.duplicated(subset=['pid', 'rfid', 'Event'])

    return features[mask], meta[mask]

def load_data(feature_keys, version, dataset, processes, indices=None, combination_type=None, selection_type=None, selection_stage=-1, use_progressbar=True):
    """Loads data corresponding to feature_keys for the given version and 
    processes.

    Returns (features, meta) where features is a dataframe containing
    branches given by feature_keys from the given processes and meta is a
    dataframe with entry ids (see cache.py for more information).

    Entries in (features, meta) are filtered according to combination_type,
    selection_type and selection_stage.

    Parameters
    ----------
    feature_keys: array_like
        The keys for each feature that should be included.

    version: one of process.VERSIONS
        The CEPC version to use.

    dataset: string
        The name of the job options to use.

    processes: array_like
        The processes that should be included.

    indices: boolean array, optional
        If set, only return data from events corresponding to the given indices.

    combination_type: one of COMBINATION_TYPES or None, optional (default='highest_pt')
        Method by which to select a single combination for each event. If
        'highest_pt', select the combination that involves the highest pT final
        state particles. If 'highest_E', select the combination that involves the highest energy final state particles. If 'ryuta', filter events without 2 jets
        and select the combination with the highest pT muons.

    selection_type: one of SELECTION_TYPES or None, optional (default=None)
        Method by which to select events. See get_selection() for details.

    selection_stage: int, optional (default=-1)
        The stage of the selection to use. See apply_selection() for details.

    use_progressbar: bool, optional (default=True)
        If set, write a textual progressbar to standard output to indicate
        loading progress.

    """

    # Validate input.
    validate.check_version(version)
    validate.check_dataset(version, dataset)
    assert combination_type in COMBINATION_TYPES or combination_type is None, 'combination_type={}'.format(combination_type)
    assert selection_type in SELECTION_TYPES or selection_type is None, 'selection_type={}'.format(selection_type)

    # Load data from cache.
    features, meta = cache.load_data(feature_keys, version, dataset, processes, use_progressbar=use_progressbar)
    logging.debug('Total number of combinations: {}'.format(meta.shape[0]))

    if features.shape[0] == 0:
        return features, meta

    # Apply combination_type filtering.
    if combination_type == 'ryuta':
        features, meta = select_ryuta_combinations(features, meta, version, dataset, use_progressbar=use_progressbar)
    elif combination_type == 'highest_pt':
        features, meta = select_highest_pt_combinations(features, meta, version, dataset, use_progressbar=use_progressbar)
    elif combination_type == 'highest_E':
        features, meta = select_highest_E_combinations(features, meta, version, dataset, use_progressbar=use_progressbar)
    logging.debug('Total number of events: {}'.format(meta.shape[0]))
    features.reset_index(drop=True, inplace=True)
    meta.reset_index(drop=True, inplace=True)

    if features.shape[0] == 0:
        return features, meta

    # Apply selection_type filtering.
    if selection_type:
        cuts = get_selection(meta['pid'], version, dataset, selection_type=selection_type, combination_type=combination_type, use_progressbar=use_progressbar)
        features, meta = apply_selection([features, meta], cuts, selection_stage)

    # Filter by given indices.
    if indices is not None:
        features = features.iloc[indices]
        meta = meta.iloc[indices]

    # Reindex
    features.reset_index(inplace=True, drop=True)
    meta.reset_index(inplace=True, drop=True)

    return features, meta

def load_train_test_data_from_run_dir(run_dir, data_type, get_indices=False, use_progressbar=True):
    """Load data from the given run_dir using the saved run.ini configuration."""
    if data_type not in ('train', 'test', 'both'):
        raise ValueError('data_type must be in ("train", "test", "both") but was {}'.format(data_type))
    config = utils.load_run_config(run_dir)
    version = config['DATA']['version']
    dataset = config['DATA']['dataset']
    processes = utils.str_to_list(config['DATA']['processes']) + [process.get_signal_process(version, dataset)]
    feature_keys = utils.str_to_list(config['DATA']['feature_keys'])
    train_test_seed = config['DATA'].getint('train_test_seed')
    combination_type = config['DATA']['combination_type']
    selection_type = config['DATA']['selection_type']
    if selection_type == 'None':
        selection_type = None
    selection_stage = config['DATA']['selection_stage']
    if selection_stage == 'None':
        selection_stage = -1
    else:
        selection_stage = int(selection_stage)
    X, y = load_data(feature_keys, version, dataset, processes, combination_type=combination_type, selection_type=selection_type, selection_stage=selection_stage, use_progressbar=use_progressbar)
    y = y['pid'].values
    skf = StratifiedKFold(n_splits=2, random_state=train_test_seed, shuffle=True)
    train_index, test_index = [x for x in skf.split(X, y)][0]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if data_type == 'both':
        if get_indices:
            return X_train, X_test, y_train, y_test, train_index, test_index
        else:
            return X_train, X_test, y_train, y_test
    elif data_type == 'train':
        if get_indices:
            return X_train, y_train, train_index
        else:
            return X_train, y_train
    elif data_type == 'test':
        if get_indices:
            return X_test, y_test, test_index
        else:
            return X_test, y_test

def load_model(run_dir):
    config = utils.load_run_config(run_dir)
    return joblib.load(os.path.join(run_dir, 'model.pkl'))

def main():
    pass

if __name__ == '__main__':
    main()
