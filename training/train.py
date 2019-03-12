"""Train

This module contains methods to train and store a machine learning model.

"""

import os
import shutil
from configparser import ConfigParser

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import utils
from config import settings
from data import load, process

RUN_DIR = settings.get_setting('Event Selection', 'run_dir')


def read_input_config(config_file_path):
    config = ConfigParser()
    config.read(config_file_path)
    version = config['DATA']['version']
    processes = utils.str_to_list(config['DATA']['processes'])
    feature_keys = utils.str_to_list(config['DATA']['feature_keys'])
    return version, processes, feature_keys


def build_bdt():
    model = GradientBoostingClassifier(verbose=1)
    return model


def build_dt():
    model = DecisionTreeClassifier()
    return model


def train(X_train, y_train, model, weights=None):
    print('[Train] Training model...')
    if weights is not None:
        model.fit(X=X_train, y=y_train, model__sample_weight=weights)
    else:
        model.fit(X=X_train, y=y_train)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train model using given features and files.')
    parser.add_argument(
        'config', help='The config file in which features and files are specified.')
    parser.add_argument('model', default='bdt', choices=[
                        'bdt', 'dt'], help='The model to use.')
    parser.add_argument('--dataset', '-d', required=True,
                        help='The dataset to use.')
    parser.add_argument('--combination_type', '-c', default='highest_pt',
                        choices=load.COMBINATION_TYPES, help='Combination type to use (see load.load_data).')
    parser.add_argument('--selection_type', '-s', default=None,
                        choices=load.SELECTION_TYPES, help='Selection type to use (see load.load_data).')
    parser.add_argument('--selection_stage', '-ss', default=None,
                        type=int, help='Selection stage to use (see load.load_data).')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--clean', action='store_true',
                        help='If set, clean the run_dir directory before saving new load.')
    args = parser.parse_args()
    if args.clean:
        try:
            shutil.rmtree(RUN_DIR)
        except Exception:
            pass
    if args.model == 'bdt':
        model = build_bdt()
        model = Pipeline(
            steps=[('scaler', StandardScaler()), ('model', model)])
    elif args.model == 'dt':
        model = build_dt()
    run_dir = utils.make_run_dir()
    print('[Train] Results will be stored in {}'.format(run_dir))
    version, processes, feature_keys = read_input_config(args.config)
    load.update_run_config(version, args.dataset, processes, feature_keys, model=args.model, run_dir=run_dir,
                           combination_type=args.combination_type, selection_type=args.selection_type, selection_stage=args.selection_stage)
    X_train, y_train = load.load_train_test_data_from_run_dir(
        run_dir, data_type='train')
    y_binary = utils.get_binary_labels(y_train, version, args.dataset)
    if args.use_weights:
        weights = np.ones(y_train.shape[0])
        for pid in np.unique(y_train):
            p = process.get_pid_map(version, args.dataset)[pid]
            weights[y_train == pid] = process.calculate_weight(
                version, args.dataset, p)
        train(X_train, y_binary, model, weights=weights)
    else:
        train(X_train, y_binary, model)
    joblib.dump(model, os.path.join(run_dir, 'model.pkl'))


if __name__ == '__main__':
    main()
