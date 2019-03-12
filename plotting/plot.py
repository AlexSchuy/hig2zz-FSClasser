import itertools
import math
import os
from collections import OrderedDict, defaultdict

import matplotlib
import numpy as np
import pandas as pd
import uproot
from matplotlib import pyplot as plt
from progressbar import progressbar

from common import utils
from config import settings
from data import load, process

PLOT_DIR = settings.get_setting('Event Selection', 'plot_dir')
default_logx = set()
default_logy = set()
units = {'JetPtP1': 'GeV', 'JetPtP2': 'GeV', 'PfoPtP3': 'GeV', 'PfoPtP4': 'GeV', 'RMass12': 'GeV', 'RMass34': 'GeV', 'RMass1234': 'GeV', 'MissingMass': 'GeV', 'TotalP': 'GeV', 'LeadingMuonPt': 'GeV', 'SubLeadingMuonPt': 'GeV', 'LeadingMuonPhi': 'rad', 'SubLeadingMuonPhi': 'rad', 'LeadingJetPt': 'GeV', 'SubLeadingJetPt': 'GeV',
         'LeadingJetPhi': 'rad', 'SubLeadingJetPhi': 'rad', 'NonDiMuonVisMass': 'GeV', 'VisPt': 'GeV', 'MinA1': 'rad', 'Rreco34': 'GeV', 'JetEnP1': 'GeV', 'JetEnP2': 'GeV', 'PfoEnP3': 'GeV', 'PfoEnP4': 'GeV', 'LeadingEnJetEn': 'GeV', 'SubLeadingEnJetEn': 'GeV', 'LeadingEnMuonEn': 'GeV', 'SubLeadinEnMuonEn': 'GeV'}
default_ranges = {'JetPtP1': (0, 70), 'JetPtP2': (0, 70), 'PfoPtP3': (0, 80), 'PfoPtP4': (0, 80), 'RMass12': (0, 125), 'RMass34': (0, 100), 'RMass1234': (0, 150), 'MissingMass': (0, 200), 'TotalP': (20, 100), 'nPFOs': (0, 60), 'LeadingMuonPt': (0, 80), 'SubLeadingMuonPt': (0, 60), 'LeadingMuonCosTheta': (-1, 1), 'SubLeadingMuonCosTheta': (-1, 1), 'LeadingMuonPhi': (0, 2 * math.pi), 'SubLeadingMuonPhi': (0, 2 * math.pi), 'LeadingJetPt': (0, 80), 'SubLeadingJetPt': (0, 50), 'LeadingJetCosTheta': (-1, 1), 'SubLeadingJetCosTheta': (-1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1), 'LeadingJetPhi': (0, 2 * math.pi), 'SubLeadingJetPhi': (0, 2 * math.pi), 'LeadingJetnPFO': (0, 100), 'SubLeadingJetnPFO': (0, 100), 'PtScalarSum': (0, 150), 'EventMultiplicity': (0, 5), 'Rreco34': (80, 240), 'MinA1': (0, math.pi), 'MinA2': (0, math.pi), 'Anlj': (0, math.pi), 'CosTheta': (-1, 1), 'JetEnP1': (0, 250), 'JetEnP2': (0, 250), 'PfoEnP3': (0, 250), 'PfoEnP4': (0, 250), 'LeadingEnJetEn': (0, 250), 'SubLeadingEnJetEn': (0, 250), 'LeadingEnMuonEn': (0, 250), 'SubLeadingEnMuonEn': (0, 250)}
default_titles = {'JetPtP1': 'pT(j1)', 'JetPtP2': 'pT(j2)', 'PfoPtP3': 'pT(m+)', 'PfoPtP4': 'pT(m-)', 'RMass12': 'M(jj)', 'RMass34': 'M(mm)', 'RMass1234': 'M(mmjj)',
                  'MissingMass2': 'Missing mass squared', 'TotalP': 'p(mmjj)', 'nPFOs': 'number of particles', 'Rreco34': 'Recoil M(mm)', 'Rreco1234': 'Recoil M(mmjj)', 'Anlj': 'lab angle(dimuon, dijet)'}


def make_2d_hist(data_x, data_y, filepath, weights=None, title=None, xlabel=None, ylabel=None, bins=40, cmap=plt.cm.Blues, **kwargs):
    for name in data_x:
        x = data_x[name]
        y = data_y[name]
        plt.hist2d(x, y, weights=weights[name], bins=bins, cmap=cmap, **kwargs)
    plt.colorbar()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    elif xlabel is not None and ylabel is not None:
        plt.title('{} vs. {}'.format(ylabel, xlabel))
    utils.makedirs(os.path.split(filepath)[0])
    plt.savefig(filepath)
    plt.clf()


def make_hist(X, filepath, weights=None, ranges=None, bins=60, label_func=None, title=None, ylabel='count', xlabel=None, logy=False, logx=None, fc=None, ec=None, hatch=None, feature_key=None, histtype='step', errorbars=False, figsize=None, fontsize=None, ylim=None, **kwargs):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if fontsize is not None:
        matplotlib.rcParams.update({'font.size': fontsize})
    plt.margins(x=0)
    if ylim is not None:
        plt.ylim(ylim)
    if weights is None:
        weights = {k: np.ones(v.shape[0]) for k, v in X.items()}
    if ranges is None:
        if feature_key in default_ranges:
            ranges = default_ranges[feature_key]
    if label_func is None:
        def label_func(s): return s
    if title is None:
        if feature_key in default_titles:
            title = '{} ({})'.format(default_titles[feature_key], feature_key)
        else:
            title = feature_key
    if xlabel is None:
        if feature_key in units:
            xlabel = '[{}]'.format(units[feature_key])
    if logy is None:
        logy = feature_key in default_logy
    if logx is None:
        logx = feature_key in default_logx
    colors = plt.cm.Paired(np.linspace(0, 1, len(X)))
    if logx:
        plt.xscale('log')
        bins = np.logspace(np.log10(ranges[0]), np.log10(
            ranges[1]), num=bins, endpoint=False)
    if histtype != 'bar':
        if ec is None:
            ec = {name: (colors[i][0], colors[i][1], colors[i][2])
                  for i, name in enumerate(X.keys())}
        if fc is None:
            fc = {name: (colors[i][0], colors[i][1], colors[i][2])
                  for i, name in enumerate(X.keys())}
        if hatch is None:
            hatch = defaultdict(lambda: None)
        for name, x in X.items():
            if x is None:
                continue
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(x, pd.DataFrame):
                x = x.values[:, 0]
            n, bin_edges, _ = plt.hist(x, bins=bins, range=ranges, histtype=histtype, label=label_func(
                name), weights=weights[name], fc=fc[name], hatch=hatch[name], ec=ec[name], **kwargs)
            if errorbars:
                # Calculate error for each bin assuming Poisson statistics.
                bin_errors = np.histogram(
                    x, weights=weights[name]**2, bins=bin_edges)[0]**(1/2)
                midpoints = 0.5*(bin_edges[1:] + bin_edges[:-1])
                plt.errorbar(midpoints, n, yerr=bin_errors,
                             fmt='none', ecolor=fc[name], capsize=4)
    else:
        hist_x = []
        hist_labels = []
        hist_weights = []
        for name, x in X.items():
            hist_x.append(x)
            hist_labels.append(label_func(name))
            hist_weights.append(weights[name])
        plt.hist(hist_x, bins=bins, range=ranges, histtype='bar', stacked=True,
                 label=hist_labels, weights=hist_weights, fc=fc, hatch=hatch, ec=ec, **kwargs)
    plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if logy:
        plt.yscale('log')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title(title)
    utils.makedirs(os.path.split(filepath)[0])
    plt.savefig(filepath)
    plt.clf()


def make_yuqian_style_hist(X, filepath, **kwargs):
    hatch = {'signal': None, 'background': '//'}
    fc = {'signal': (0.59, 0.64, 0.82, 1), 'background': (0, 0, 0, 0)}
    ec = {'signal': (0, 0, 1, 1),  'background': (1, 0, 0, 1)}
    make_hist(X, filepath, hatch=hatch, fc=fc, ec=ec,
              lw=2, histtype='stepfilled', **kwargs)


def make_standard_hist(version, dataset, processes, feature_keys, weight_type, group_type, combination_type='highest_pt', selection_type=None, selection_stage=-1, indices=None, test_fraction=1.0, plot_dir=None, **kwargs):
    if isinstance(feature_keys, str):
        feature_keys = [feature_keys]

    # Get data and separate it into appropriate groups.
    df, meta = load.load_data(feature_keys, version, dataset, processes, indices=indices, combination_type=combination_type,
                              selection_type=selection_type, selection_stage=selection_stage, use_progressbar=False)
    groups = process.get_groups(version, dataset, group_type)
    features, weights = process.by_group(version, dataset, groups, meta['pid'], [
                                         df], weight_type=weight_type, test_fraction=test_fraction)

    for feature_key in feature_keys:
        # Make plots.
        if plot_dir is None:
            weight_name = weight_type + '_weight'
            group_name = group_type + '_group'
            if selection_type is None:
                selection_name = 'no_selection'
            else:
                selection_name = str(selection_type) + '_selection'
            if selection_stage == -1:
                filepath = os.path.join(
                    PLOT_DIR, version, dataset, weight_name, group_name, selection_name, feature_key)
            else:
                selection_stage_name = 'cut_' + str(selection_stage)
                filepath = os.path.join(PLOT_DIR, version, dataset, weight_name,
                                        group_name, selection_name, selection_stage_name, feature_key)
        else:
            filepath = os.path.join(plot_dir, feature_key)
        ylabel = '{} count'.format(weight_type)
        if group_type == 'all_background':
            make_yuqian_style_hist(OrderedDict((k, features[k][feature_key]) for k in features if features[k]
                                               is not None), filepath, weights=weights, ylabel=ylabel, feature_key=feature_key, **kwargs)
        else:
            make_hist(OrderedDict((k, features[k][feature_key]) for k in features if features[k] is not None),
                      filepath, weights=weights, ylabel=ylabel, feature_key=feature_key, histtype='step', **kwargs)


def make_standard_2d_hist(version, dataset, processes, feature_key_x, feature_key_y, weight_type, group_type, combination_type='highest_pt', selection_type=None, selection_stage=-1, indices=None, test_fraction=1.0, filepath=None, histtype='step', **kwargs):

    # Get data and separate it into appropriate groups.
    df, meta = load.load_data([feature_key_x, feature_key_y], version, dataset, processes, indices=indices,
                              combination_type=combination_type, selection_type=selection_type, selection_stage=selection_stage, use_progressbar=False)
    groups = process.get_groups(version, dataset, group_type)
    features, weights = process.by_group(version, dataset, groups, meta['pid'], [
                                         df], weight_type=weight_type, test_fraction=test_fraction)
    # Make plots.
    if filepath is None:
        filepath = os.path.join(PLOT_DIR, version, dataset, weight_type, group_type, str(
            selection_type), str(combination_type), '{}_vs_{}'.format(feature_key_y, feature_key_x))
    features_x = {k: v[feature_key_x]
                  for k, v in features.items() if v is not None}
    features_y = {k: v[feature_key_y]
                  for k, v in features.items() if v is not None}
    make_2d_hist(features_x, features_y, filepath, weights=weights,
                 ylabel=feature_key_y, xlabel=feature_key_x, **kwargs)


def plot_input(version, dataset, processes=None, **kwargs):
    if processes is None:
        processes = process.get_all_processes(version, dataset)
    feature_keys = ['RMass12', 'RMass34', 'Rreco34', 'Rreco1234', 'Anlj', 'TotalP', 'CosTheta',
                    'nPFOs', 'RMass1234', 'LeadingEnJetEn', 'SubLeadingEnJetEn', 'VisMass', 'MissingMass']
    make_standard_hist(version, dataset, processes, feature_keys, **kwargs)


def make_cross_check_plots():
    feature_keys = ['InvisMass', 'RMass34', 'Rreco34',
                    'nPFOs', 'VisPt', 'MinA1Lab', 'RMass12']
    ranges = {'InvisMass': (0, 250), 'RMass12': (0, 250), 'RMass34': (0, 250), 'Rreco34': (
        0, 250), 'nPFOs': (0, 100), 'VisPt': (0, 150), 'MinA1Lab': (0, 2.62), 'NonDiMuonVisMass': (0, 250)}
    bins = {'InvisMass': 250, 'RMass12': 250, 'RMass34': 250, 'Rreco34': 250,
            'nPFOs': 100, 'VisPt': 200, 'MinA1Lab': 30, 'NonDiMuonVisMass': 200}

    # Plot selection variables after previous selections.
    for i, feature_key in enumerate(feature_keys):
        filepath = os.path.join(PLOT_DIR, 'cross_check', feature_key)
        logy = feature_key != 'MinA1Lab'
        make_standard_hist('v4', ['llh_zz'], feature_key, weight_type='expected', group_type='ungrouped', selection_type='ryuta', selection_stage=i, combination_type='ryuta', filepath=filepath, fc={'llh_zz': (1, 0, 0, 1)}, hatch={
                           'llh_zz': None}, ec={'llh_zz': (0, 0, 0, 1)}, logy=logy, ranges=ranges[feature_key], figsize=(16, 7), fontsize=22, histtype='stepfilled', bins=bins[feature_key], dataset='cross_check')

    # Plot the final M(mm)
    filepath = os.path.join(PLOT_DIR, 'cross_check', 'RMass12Final')
    make_standard_hist('v4', ['llh_zz'], 'RMass34', weight_type='expected', group_type='ungrouped', selection_type='ryuta', combination_type='ryuta', filepath=filepath, fc={'llh_zz': (
        1, 0, 0, 1)}, hatch={'llh_zz': None}, ec={'llh_zz': (0, 0, 0, 1)}, logy=False, ranges=(75, 105), figsize=(16, 7), fontsize=22, histtype='stepfilled', dataset='cross_check')


def make_cross_check_preselection_plots():
    feature_keys = ['InvisMass', 'MissingMass', 'RMass12', 'RMass34']
    ranges = {'InvisMass': (-50, 250), 'MissingMass': (-50, 250),
              'RMass12': (0, 250), 'RMass34': (0, 250)}
    for feature_key in feature_keys:
        filepath = os.path.join(
            PLOT_DIR, 'cross_check', 'after_preselection', feature_key)
        make_standard_hist('v4', ['llh_zz'], feature_key, weight_type='expected', group_type='ungrouped', selection_type=None, combination_type='ryuta', filepath=filepath, fc={'llh_zz': (
            1, 0, 0, 1)}, hatch={'llh_zz': None}, ec={'llh_zz': (0, 0, 0, 1)}, logy=False, ranges=ranges[feature_key], figsize=(16, 7), fontsize=22, bins=250, dataset='cross_check')

    filepath = os.path.join(PLOT_DIR, 'cross_check',
                            'after_preselection', 'RMass12_vs_InvisMass')
    make_standard_2d_hist('v4', ['llh_zz'], 'RMass12', 'InvisMass', weight_type='expected', group_type='ungrouped',
                          selection_type=None, combination_type='ryuta', filepath=filepath, bins=200, dataset='cross_check')


def make_truth_study_plots(datasets):
    feature_keys = ['mc_zmm_m', 'mc_zqq_m', 'dimuon_m', 'dijet_m', 'mc_q1_e',
                    'mc_q2_e', 'mc_mu1_e', 'mc_mu2_e', 'n_muon', 'n_electron', 'iso_lep_non_dimuon_e']
    ranges = {'mc_zmm_m': (0, 250), 'mc_zqq_m': (0, 250), 'dimuon_m': (0, 250), 'dijet_m': (0, 250), 'mc_q1_e': (0, 250), 'mc_q2_e': (
        0, 250), 'mc_mu1_e': (0, 100), 'mc_mu2_e': (0, 50), 'n_muon': (0, 10), 'n_electron': (0, 10), 'iso_lep_non_dimuon_e': (0, 100)}
    ylims = {'mc_zmm_m': 30, 'mc_zqq_m': 30, 'dimuon_m': 30, 'dijet_m': 30}

    for dataset in datasets:
        features, _ = load.load_data(['mc_zz_flag', 'is_signal', 'is_mumujj'], 'v4', dataset, [
                                     'nnh_zz'], combination_type=None)
        signal_indices = np.where(features['is_signal'])[0]
        reco_indices = np.where(features['is_mumujj'])[0]
        reco_signal_indices = np.intersect1d(reco_indices, signal_indices)
        reco_background_indices = np.setdiff1d(reco_indices, signal_indices)

        # Compare reco signal and background data sets.
        reco_features, _ = load.load_data(['dijet_m', 'dimuon_m', 'n_col_reco'], 'v4', dataset, [
                                          'nnh_zz'], combination_type=None)
        for feature_key in ['dijet_m', 'dimuon_m', 'n_col_reco']:
            filepath = os.path.join(
                PLOT_DIR, dataset, 'reco_signal_vs_background', feature_key)
            X = {'signal': reco_features[feature_key].loc[reco_signal_indices],
                 'background': reco_features[feature_key].loc[reco_background_indices]}
            make_hist(X, filepath=filepath, feature_key=feature_key,
                      histtype='step', ec={'signal': 'g', 'background': 'r'})

        # Plot each of the data sets.
        for set_name, indices in zip(('reco', 'signal', 'reco_signal'), (reco_indices, signal_indices, reco_signal_indices)):

            for feature_key in feature_keys:
                filepath = os.path.join(
                    PLOT_DIR, dataset, set_name, feature_key)
                if feature_key in ylims:
                    ylim = (0, ylims[feature_key])
                else:
                    ylim = None
                make_standard_hist('v4', dataset, ['nnh_zz'], feature_key, indices=indices, weight_type='expected', group_type='ungrouped', selection_type=None, combination_type=None, filepath=filepath, fc={
                                   'nnh_zz': (1, 0, 0, 1)}, hatch={'nnh_zz': None}, ec={'nnh_zz': (0, 0, 0, 1)}, logy=False, ranges=ranges[feature_key], bins=250, ylim=ylim)

            for second_indices, x, y in zip((signal_indices, reco_indices), ('mc_zmm_m', 'dimuon_m'), ('mc_zqq_m', 'dijet_m')):
                filepath = os.path.join(
                    PLOT_DIR, dataset, set_name, '{}_vs_{}'.format(x, y))
                make_standard_2d_hist('v4', dataset, ['nnh_zz'], x, y, indices=np.intersect1d(
                    indices, second_indices), weight_type='expected', group_type='ungrouped', selection_type=None, combination_type=None, filepath=filepath, bins=20)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Make plots.')
    subparsers = parser.add_subparsers(
        dest='function', help='Plot function to run.')

    # input features
    parser_features = subparsers.add_parser(
        'input_features', help='Plot the features that are used as inputs for analysis.')
    parser_features.add_argument('--version', '-v', default='v4',
                                 choices=process.VERSIONS, help='The CEPC version to use.')
    parser_features.add_argument('--processes', '-p', nargs='+', default=None,
                                 help='The processes to plot. Otherwise, plot all processes.')
    parser_features.add_argument(
        '--weight', '-w', default='normalized', choices=process.WEIGHT_TYPES)
    parser_features.add_argument('--group', '-g', default='sl_separated', choices=process.GROUP_TYPES,
                                 help='The way individual processes should be grouped for the table (see process module for details).')
    parser_features.add_argument('--combination', '-c', default=None,
                                 choices=load.COMBINATION_TYPES, help='combination type to use (see load.load_data).')
    parser_features.add_argument('--selection', '-s', default=None,
                                 choices=load.SELECTION_TYPES, help='Selection type to use (see load.load_data).')
    parser_features.add_argument('--selection_stage', '-ss', default=-1,
                                 type=int, help='Selection stage to use (see load.load_data).')
    parser_features.add_argument(
        'dataset', help='The name of the data set to use.')

    # cross-check features.
    parser_cross_check_features = subparsers.add_parser(
        'cross_check_features', help='Make plots of features used by Ryuta in the llh_zz analysis.')

    # cross-check pre-selection comparison.
    parser_cross_check_preselection = subparsers.add_parser(
        'cross_check_preselection', help='Make plots of features used by Ryuta in the llh_zz analysis after preselection.')

    # truth study.
    parser_truth_study = subparsers.add_parser(
        'truth_study', help='Make plots of MC truth information.')
    parser_truth_study.add_argument(
        'datasets', nargs='+', help='datasets to use.')

    args = parser.parse_args()
    if args.function == 'input_features':
        plot_input(version=args.version, processes=args.processes, weight_type=args.weight, group_type=args.group,
                   combination_type=args.combination, selection_type=args.selection, selection_stage=args.selection_stage, dataset=args.dataset)
    elif args.function == 'cross_check_features':
        make_cross_check_plots()
    elif args.function == 'cross_check_preselection':
        make_cross_check_preselection_plots()
    elif args.function == 'truth_study':
        make_truth_study_plots(args.datasets)


if __name__ == '__main__':
    main()
