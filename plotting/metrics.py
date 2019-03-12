"""Metrics

This module contains methods to analyze the effectiveness of the selection
procedures.

"""

import argparse
import os
from collections import OrderedDict

import graphviz
import numpy as np
import pandas as pd
from sklearn import tree

from common import utils, validate
from config import settings
from data import load, process
from plotting import plot

RUN_DIR = settings.get_setting('Event Selection', 'run_dir')


def cut_table(version, dataset, group_type, combination_type, selection_type):
    groups = process.get_groups(version, dataset, group_type)

    def count(p, meta):
        return np.sum(meta['pid'] == process.get_pid_map(version, dataset)[p])
    processes = process.get_all_processes(version, dataset)
    features, meta = load.load_data(
        ['RMass12'], version, dataset, processes, combination_type=combination_type)
    cuts = load.get_selection(meta['pid'], version, dataset,
                              combination_type=combination_type, selection_type=selection_type)
    index = np.concatenate([['initial', 'final_state_selection'],
                            cuts.columns.values, ['sigma', 'efficiency']])
    raw_cut_df = pd.DataFrame(index=index, columns=processes)
    group_cut_df = pd.DataFrame(index=index, columns=groups.keys())
    for p in processes:
        raw_cut_df.loc['initial', p] = process.get_num_MC_events(
            version, dataset, p, test_fraction=0.5)
        raw_cut_df.loc['final_state_selection', p] = count(p, meta)
    for cut in cuts:
        meta = meta[cuts[cut]]
        cuts = cuts[cuts[cut]]
        for p in processes:
            raw_cut_df.loc[cut, p] = count(p, meta)
    for k, v in groups.items():
        w = np.array([process.calculate_weight(version, dataset, p, test_fraction=0.5)
                      for p in v])
        if cuts.shape[0] > 0:
            n_f = np.array([raw_cut_df.loc[cuts.columns[-1], p] for p in v])
        else:
            n_f = np.array(
                [raw_cut_df.loc['final_state_selection', p] for p in v])
        group_cut_df.loc['sigma', k] = np.sum(w**2*n_f)**(1/2)
    count_index = np.concatenate(
        [['final_state_selection'], cuts.columns.values])
    for k, v in groups.items():
        group_cut_df.loc[count_index, k] = np.sum([process.calculate_weight(
            version, dataset, p, test_fraction=0.5)*raw_cut_df.loc[count_index, p] for p in v], axis=0)
        group_cut_df.loc['initial', k] = np.sum(
            [process.get_num_expected_events(version, dataset, p) for p in v], axis=0)
    if cuts.shape[0] > 0:
        group_cut_df.loc['efficiency', :] = group_cut_df.loc[cuts.columns[-1],
                                                             :] / group_cut_df.loc['initial', :]
    else:
        group_cut_df.loc['efficiency', :] = 1.0
    return group_cut_df


def analyze_standard(run_dir):
    """Make basic bdt analysis plots."""

    # Load run data
    model = load.load_model(run_dir)
    X_test, y_test, test_index = load.load_train_test_data_from_run_dir(
        run_dir, 'test', get_indices=True)
    config = utils.load_run_config(run_dir)
    model_name = config['DATA']['model']
    dataset = config['DATA']['dataset']
    version = config['DATA']['version']
    feature_keys = utils.str_to_list(config['DATA']['feature_keys'])
    processes = utils.str_to_list(
        config['DATA']['processes']) + [process.get_signal_process(version, dataset)]
    combination_type = config['DATA']['combination_type']
    selection_type = config['DATA']['selection_type']
    if selection_type == 'None':
        selection_type = None
    selection_stage = config['DATA']['selection_stage']
    if selection_stage == 'None':
        selection_stage = -1
    else:
        selection_stage = int(selection_stage)
    y_test_binary = utils.get_binary_labels(y_test, version, dataset)
    y_expected = model.predict_proba(X_test)[:, 1]

    if model_name == 'bdt':
        # Plot BDT response
        X, weights = process.by_group(version, dataset, process.get_signal_background_groups(
            version, dataset), y_test, [y_expected], weight_type='normalized', test_fraction=0.5)
        feature_key = 'BDT_response'
        filepath = os.path.join(run_dir, 'plots', feature_key)
        plot.make_yuqian_style_hist(X, filepath, weights=weights, ranges=(
            0.0, 1.0), feature_key=feature_key, ylabel='normalized count')

        # Plot BDT response for selected events
        X, weights = process.by_group(version, dataset, process.get_signal_background_groups(
            version, dataset), y_test[y_expected > 0.5], [y_expected[y_expected > 0.5]], weight_type='expected', test_fraction=0.5)
        filepath = os.path.join(
            run_dir, 'plots', 'BDT_response (selected events)')
        plot.make_yuqian_style_hist(X, filepath, weights=weights, ranges=(
            0.5, 1.0), feature_key=feature_key, ylabel='expected count')
    elif model_name == 'dt':
        # Show dt
        dot_data = tree.export_graphviz(model, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render('nnh_zz_mmjj', view=True)

    # Plot feature variables for selected events
    plot.make_standard_hist(version=version, processes=processes, feature_keys=feature_keys, weight_type='expected', group_type='all_background',
                            indices=test_index[y_expected > 0.5], combination_type=combination_type, selection_type=selection_type, selection_stage=selection_stage, dataset=dataset, plot_dir=os.path.join(run_dir, 'plots', 'expected', 'all_background'))

    print('[metrics] feature importance:')
    feature_importance = dict(
        zip(X_test.columns.values, model.named_steps['model'].feature_importances_))
    for feature_name, importance in sorted(feature_importance.items(), key=lambda kv: -kv[1]):
        print('{}: {}'.format(feature_name, importance))
    y_selected = y_test[y_expected > 0.5]
    show_surviving_processes(y_selected, version, dataset)


def show_surviving_processes(y_selected, version, dataset):
    IDs, counts = np.unique(y_selected, return_counts=True)
    print('Selected samples by process (weighted):')
    for ID, count in zip(IDs, counts):
        p = process.get_pid_map(version, dataset)[ID]
        weight = process.calculate_weight(
            version, dataset, p, test_fraction=0.5)
        print('\t{}: {} ({})'.format(p, count, count * weight))


def analyze_truth_study(datasets):
    passed_events = pd.DataFrame(
        columns=['mc', 'reco', 'both', 'any'], index=datasets)
    reco_zz_distribution = pd.DataFrame(
        columns=[11, 12, 13, 21, 22, 23, 31, 32, 33], index=datasets)
    for dataset in datasets:
        features, _ = load.load_data(['mc_zz_flag', 'is_signal', 'is_mumujj'], 'v4', dataset, [
                                     'nnh_zz'], combination_type=None)
        passed_events.loc[dataset, 'mc'] = np.sum(features['is_signal'])
        passed_events.loc[dataset, 'reco'] = np.sum(features['is_mumujj'])
        passed_events.loc[dataset, 'both'] = np.sum(
            np.logical_and(features['is_signal'], features['is_mumujj']))
        passed_events.loc[dataset, 'any'] = np.sum(
            np.logical_or(features['is_signal'], features['is_mumujj']))

        reco_indices = np.where(features['is_mumujj'])[0]
        flags, counts = np.unique(
            features.loc[reco_indices]['mc_zz_flag'], return_counts=True)
        for flag, count in zip(flags, counts):
            reco_zz_distribution.loc[dataset, flag] = count
        for flag in [f for f in reco_zz_distribution.columns if f not in flags]:
            reco_zz_distribution.loc[dataset, flag] = 0

    print('Event distribution:')
    print(passed_events)
    print('-----')
    print('reco zz distribution:')
    print(reco_zz_distribution)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate analysis performance.')
    subparsers = parser.add_subparsers(
        dest='function', help='Metric function to use.')

    # cut_flow
    parser_cf = subparsers.add_parser(
        'cutflow', help='Print a cut-flow table for the given selections.')
    parser_cf.add_argument('--combination', '-c', default='highest_pt',
                           choices=load.COMBINATION_TYPES, help='combination type to use (see load.load_data).')
    parser_cf.add_argument('--selection', '-s', default=None, choices=load.SELECTION_TYPES,
                           help='Selection type to use (see load.load_data).')
    parser_cf.add_argument('--version', '-v', default='v4',
                           choices=process.VERSIONS, help='The CEPC version to use.')
    parser_cf.add_argument('--group_type', '-g', default='sl_separated', choices=process.GROUP_TYPES,
                           help='The way individual processes should be grouped for the table (see process module for details).')
    parser_cf.add_argument('--dataset', '-d', required=True,
                           help='The name of the job options file that was used when processing the load.')

    # analyze bdt/dt
    parser_standard = subparsers.add_parser(
        'standard', help='Analyze the results of training.')
    parser_standard.add_argument(
        '--run_dir', default=None, help='The run dir to use.')

    # truth study
    parser_ts = subparsers.add_parser(
        'truth_study', help='Print truth study information')
    parser_ts.add_argument('--dataset', '-d', required=True,
                           nargs='+', help='datasets to use.')

    args = parser.parse_args()

    if args.function == 'cutflow':
        print(cut_table(args.version, args.dataset,
                        args.group_type, args.combination, args.selection))
    elif args.function == 'standard':
        if args.run_dir:
            run_dir = os.path.join(RUN_DIR, args.run_dir)
        else:
            run_dir = utils.most_recent_dir()
        analyze_standard(run_dir)
    elif args.function == 'truth_study':
        analyze_truth_study(args.dataset)


if __name__ == '__main__':
    main()
