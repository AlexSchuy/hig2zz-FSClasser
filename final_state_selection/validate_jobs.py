from common import utils
from config import settings
import os
import glob
import re
import argparse
import progressbar

VALIDATION_DIR = settings.get_setting('Final State Selection', 'validation_dir')
LOG_DIR = settings.get_setting('Final State Selection', 'log_dir')

def get_last_line(path):
    last = ''
    with open(path) as f:
        line = ''
        for line in f:
            pass
        last = line
    return last

def validate_job(log):
    last_line = get_last_line(log)
    return re.search('processed [0-9]* events', last_line) is not None

def validate_process(log_dir):
    logs = glob.glob(os.path.join(log_dir, '*'))
    jobs = [os.path.splitext(os.path.split(l)[1])[0] for l in logs]
    good_jobs = [j for j,l in zip(jobs,logs) if validate_job(l)]
    bad_jobs = [j for j,l in zip(jobs,logs) if not validate_job(l)]
    return good_jobs, bad_jobs

def validate_dataset(version, dataset):
    dataset_log_dir = os.path.join(LOG_DIR, version, dataset)
    dataset_validation_dir = os.path.join(VALIDATION_DIR, version, dataset)
    utils.makedirs(dataset_validation_dir)
    process_log_dirs = glob.glob(os.path.join(dataset_log_dir, '*'))
    result_dict = {}
    for log_dir in progressbar.progressbar(process_log_dirs, redirect_stdout=True):
        print(log_dir)
        process = os.path.split(log_dir)[1]
        good_jobs, bad_jobs = validate_process(log_dir)
        if len(good_jobs) > 0:
            good_jobs_list = os.path.join(dataset_validation_dir, f'{process}_good.txt')
            with open(good_jobs_list, 'w') as f:
                for good_job in sorted(good_jobs):
                    f.write(f'{good_job}\n')
        if len(bad_jobs) > 0:
            bad_jobs_list = os.path.join(dataset_validation_dir, f'{process}_bad.txt')
            with open(bad_jobs_list, 'w') as f:
                for bad_job in sorted(bad_jobs):
                    f.write(f'{bad_job}\n')
        result_dict[process] = (len(good_jobs), len(bad_jobs))
    
    with open(os.path.join(dataset_validation_dir, 'results.txt'), 'w') as f:
        for process in sorted(result_dict):
            n_good, n_bad = result_dict[process]
            n_total = n_good + n_bad
            status = 'GOOD' if n_bad == 0 else 'BAD'
            f.write(f'{process}: {n_good}/{n_total} (status={status})\n')


def main():
    parser = argparse.ArgumentParser(description='Validate the fs ntuples for a particular dataset.')
    parser.add_argument('dataset', help='Dataset to use.')
    parser.add_argument('--version', '-v', default='v4', help='CEPC version to use.')
    args = parser.parse_args()
    validate_dataset(args.version, args.dataset)


if __name__ == '__main__':
    main()
