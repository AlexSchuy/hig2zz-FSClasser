import math
import os
import re
import stat
import subprocess
import time
from argparse import ArgumentParser
from configparser import ConfigParser
from glob import glob

from config import settings
from common import utils

SETUP_DIR = settings.get_setting('Final State Selection', 'setup_dir')
OUTPUT_DIR = settings.get_setting('Final State Selection', 'output_dir')
LOG_DIR = settings.get_setting('Final State Selection', 'log_dir')
JOB_DIR = settings.get_setting('Final State Selection', 'job_dir')
STEERING_DIR = settings.get_setting('Final State Selection', 'steering_dir')
TEMPLATE_DIR = settings.get_setting('Final State Selection', 'template_dir')
MACHINES = settings.get_setting('Final State Selection', 'machines')
SAMPLE_TABLE = settings.get_setting('Final State Selection', 'sample_table')
NUM_EVENTS_TABLE = settings.get_setting('Final State Selection', 'num_events_table')
SETUP_TABLE = settings.get_setting('Final State Selection', 'environment_setup_table')

def exists(name, output_dir):
    return os.path.isfile(os.path.join(output_dir, name + '.root'))


def make_steering_file(name, steering_dir, output_dir, version, gen_process, num_events, input_file_names, template):
    ''' Make a steering file with the given parameters from the appropriate template. '''
    if isinstance(input_file_names, str):
        input_files_str = input_file_names
    else:
        input_files_str = ''
        for file_name in input_file_names:
            input_files_str += '    ' + file_name + '\n'
    with open(template, 'r') as f:
        data = f.read()
    data = data.replace('%LCIO_INPUT_FILES', input_files_str)
    data = data.replace('%MAX_RECORD_NUMBER', str(num_events + 1))
    data = data.replace('%OUTPUT_ROOT_FILE',
                        os.path.join(output_dir, name + '.root'))
    steering_file_name = os.path.join(steering_dir, name + '.xml')
    with open(steering_file_name, 'w+') as f:
        f.write(data)
    return steering_file_name


def run_jobs(version, gen_processes, num_jobs_list, first_jobs, hep_sub, template_name, recreate):
    template_path = os.path.join(
        TEMPLATE_DIR, version, template_name + '.xml')
    setup_config = ConfigParser()
    setup_config.read(SETUP_TABLE)
    setup_script = setup_config.get(version, template_name)
    job_queue = []
    for gen_process, num_jobs, first_job in zip(gen_processes, num_jobs_list, first_jobs):
        sample_file_names = sorted(
            glob(utils.get_samples_dict(version)[gen_process]))
        num_events_config = ConfigParser()
        num_events_config.read(NUM_EVENTS_TABLE)
        if num_events_config.has_option(version, gen_process):
            num_files_per_job = 1000 / \
                num_events_config.getint(version, gen_process)
        else:
            num_files_per_job = 1
        if num_jobs == 0:
            num_jobs = len(sample_file_names)
        num_jobs = min(num_jobs, len(sample_file_names) /
                       num_files_per_job - first_job)
        num_jobs_remaining = num_jobs
        for i in range(first_job / 10, int(math.ceil((first_job + num_jobs) / 10.0))):
            for j in range(10):
                if num_jobs_remaining == 0:
                    break
                num_jobs_remaining -= 1
                name = '{0}_{1}_{2:03d}_{3}000'.format(
                    version, gen_process, i, str(j))
                steering_dir = os.path.join(
                    STEERING_DIR, version, template_name, gen_process)
                output_dir = os.path.join(
                    OUTPUT_DIR, version, template_name, gen_process)
                utils.makedirs(steering_dir)
                utils.makedirs(output_dir)
                if exists(name, output_dir) and not recreate:
                    print('ignoring {0}: already exists.'.format(name))
                    continue
                steering_filepath = make_steering_file(name, steering_dir, output_dir, version, gen_process, 1000,
                                                       sample_file_names[num_files_per_job*(10*i+j):num_files_per_job*(10*i+j)+num_files_per_job], template_path)
                log_dir = os.path.join(
                    LOG_DIR, version, template_name, gen_process)
                utils.makedirs(log_dir)
                log_filepath = os.path.join(log_dir, name + '.log')
                job_queue.append((steering_filepath, log_filepath))
    if hep_sub:
        for steering_filepath, log_filepath in job_queue:
            job_name = os.path.splitext(os.path.basename(steering_filepath))[0]
            job_path = os.path.join(JOB_DIR, job_name+'.sh')
            with open(job_path, 'w+') as job_file:
                job_file.write('source {0}\n'.format(
                    os.path.join(SETUP_DIR, setup_script)))
                job_file.write('Marlin {0}\n'.format(steering_filepath))
            st = os.stat(job_path)
            os.chmod(job_path, st.st_mode | stat.S_IXUSR |
                     stat.S_IXGRP | stat.S_IXOTH)
            subprocess.call(
                'hep_sub -o {0} -e {0} {1}'.format(log_filepath, job_path), shell=True)

    else:
        t0 = time.time()
        procs = dict((k, None) for k in MACHINES)
        while job_queue:
            steering_filepath, log_filepath = job_queue.pop(0)
            submitted = False
            while not submitted:
                for machine in MACHINES:
                    if procs[machine] is None or procs[machine].poll() is not None:
                        arg = 'ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -t {0} "cd {1} && . {2} && Marlin {3}" > {4}'.format(
                            machine, os.getcwd(), setup_script, steering_filepath, log_filepath)
                        procs[machine] = subprocess.Popen(arg, shell=True)
                        submitted = True
                        break
                time.sleep(0.1)
            print('{0} jobs remaining in queue.'.format(len(job_queue)))
        for proc in list(procs.values()):
            if proc is not None:
                proc.wait()
        t1 = time.time()
        print('total time = {0}s'.format(str(t1 - t0)))


def main():
    parser = ArgumentParser(
        description='Run a mmjj final state classifier job with the given configuration.')
    parser.add_argument('--version', '-v', required=True,
                        choices=['v4', 'v1'], help='CEPC version to use.')
    parser.add_argument('--gen_process', '-p', nargs='+', choices=(list(utils.get_samples_dict('v1').keys()) +
                                                                   list(utils.get_samples_dict('v4').keys())), help='Generating process to use for the job.')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Use all generating processes.')
    parser.add_argument('--recreate', '-r', action='store_true',
                        help='Force replacement of existing files.')
    parser.add_argument('--num_jobs', '-n', nargs='+', required=True,
                        help='Number of jobs to process. Each job corresponds to 1000 events. A value of 0 indicates that all jobs should be processed.', type=int)
    parser.add_argument('--hep_sub', action='store_true',
                        help='Use the batch service to run this job.')
    parser.add_argument('--first_job', '-f', nargs='+',
                        type=int, default=None, help='First job to run.')
    parser.add_argument('--template', '-t', default='JO_2',
                        help='The name of the template to use.')
    args = parser.parse_args()
    if not args.first_job:
        args.first_job = [0]
    if args.all:
        args.gen_process = list(utils.get_samples_dict(args.version).keys())
    if len(args.num_jobs) != len(args.gen_process):
        if len(args.num_jobs) == 1:
            args.num_jobs = args.num_jobs * len(args.gen_process)
        else:
            raise ValueError('Inconsistent number of parameters.')
    if len(args.first_job) != len(args.gen_process):
        if len(args.first_job) == 1:
            args.first_job = args.first_job * len(args.gen_process)
        else:
            raise ValueError('Inconsistent number of parameters.')
    run_jobs(args.version, args.gen_process, args.num_jobs,
             args.first_job, args.hep_sub, args.template, args.recreate)


if __name__ == '__main__':
    main()
