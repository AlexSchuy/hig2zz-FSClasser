import argparse
import glob
import logging
import os
import subprocess
from configparser import ConfigParser

import progressbar

from common import utils
from config import settings

MERGED_INPUT_DIR = settings.get_setting('Event Selection', 'merged_fs')
INPUT_DIR = settings.get_setting('Event Selection', 'fs')
RAW_INPUT_DIR = settings.get_setting('Common', 'fs_output')


def merge_root_files(version, datasets, mc_events_per_file):
    config = ConfigParser()
    utils.makedirs(os.path.join(MERGED_INPUT_DIR, version))
    config_path = os.path.join(
        MERGED_INPUT_DIR, version, 'num_events.ini')
    config.read(config_path)
    if datasets is None:
        datasets = [os.path.split(d)[1] for d in glob.glob(
            os.path.join(INPUT_DIR, version, '*')) if os.path.isdir(d)]
    for dataset in datasets:
        logging.info('Merging {}...'.format(dataset))
        processes = [p for p in glob.glob(os.path.join(
            INPUT_DIR, version, dataset, '*')) if os.path.isdir(p)]
        config[dataset] = {}
        procs = []
        for p in processes:
            p_name = os.path.split(p)[1]
            utils.makedirs(os.path.join(
                MERGED_INPUT_DIR, version, dataset, p_name))
            merged_file_path = os.path.join(
                MERGED_INPUT_DIR, version, dataset, p_name, p_name + '.root')
            files = glob.glob(os.path.join(p, '*.root'))
            config[dataset][p_name] = str(mc_events_per_file * len(files))
            args = 'hadd -f {} {}'.format(
                merged_file_path, os.path.join(INPUT_DIR, version, dataset, p_name, '*'))
            procs.append(subprocess.Popen(args, executable='/bin/bash',
                                          shell=True, stdout=open(os.devnull, 'w')))
        # Wait for the merge operations to complete.
        with progressbar.ProgressBar(max_value=len(procs), redirect_stdout=True) as bar:
            i = 0
            while i != len(procs):
                i = 0
                for p in procs:
                    if p.poll() is not None:
                        i += 1
                bar.update(i)

    # Write the num_events config file.
    with open(config_path, 'w+') as config_file:
        config.write(config_file)


def synchronize_data(user, host, src, dest):
    cmd = 'rsync -avzh --progress {}@{}:{}/* {}'.format(user, host, src, dest)
    returncode = subprocess.Popen(cmd.split()).wait()
    if returncode != 0:
        raise Exception(
            'Data synchronization failed with error code: {}'.format(returncode))


def preprocess_data(version, datasets=None):
    if datasets is None:
        input_paths = [d for d in glob.glob(os.path.join(
            RAW_INPUT_DIR, version, '*')) if os.path.isdir(d)]
        output_paths = [os.path.join(INPUT_DIR, version, os.path.split(d)[
                                     1]) for d in input_paths]
    else:
        input_paths = [os.path.join(
            RAW_INPUT_DIR, version, dataset) for dataset in datasets]
        output_paths = [os.path.join(
            INPUT_DIR, version, dataset) for dataset in datasets]

    for input_path, output_path in zip(input_paths, output_paths):
        utils.makedirs(output_path)
        cmd = """root -q -b -x 'data/preprocessing.cpp("{}", "{}")'""".format(
            input_path, output_path)
        returncode = subprocess.Popen(
            cmd, shell=True, executable='/bin/bash').wait()
        if returncode != 0:
            raise Exception('Merging with input={} and output={} failed with errorcode: {}'.format(
                input_path, output_path, returncode))


def main():
    parser = argparse.ArgumentParser(
        description='Perform necessary synchronization, preprocessing, and merging of load.')
    parser.add_argument('function', default='all', choices=[
                        'all', 'synchronize', 'preprocess', 'merge'], help='Operation to perform.')
    parser.add_argument('--datasets', '-d', nargs='+', default=None,
                        help='datasets to use. Otherwise, use all.')
    parser.add_argument('--version', '-v', default='v4',
                        help='CEPC version to use.')
    parser.add_argument('--events_per_file', '-n', default=1000,
                        help='Number of mc events that were used to generate each file (only relevant if function is all or preprocess).')
    parser.add_argument('--usr', default='alexjschuy',
                        help='The appropriate user on the host for synchronization.')
    parser.add_argument('--host', default='lxslc610.ihep.ac.cn',
                        help='The address of the hsot for synchronization.')
    parser.add_argument('--src', default='/cefs/higgs/alexjschuy/nnh_zz_mmjj/FSClasser_output',
                        help='The path to the source data files on the host for synchronization.')
    parser.add_argument('--dest', default=RAW_INPUT_DIR,
                        help='The data destination path for synchronization.')

    args = parser.parse_args()
    if args.function in ('synchronize', 'all'):
        synchronize_data(args.usr, args.host, args.src, args.dest)
    if args.function in ('preprocess', 'all'):
        preprocess_data(args.version, args.datasets)
    if args.function in ('merge', 'all'):
        merge_root_files(args.version, args.datasets, args.events_per_file)


if __name__ == '__main__':
    main()
