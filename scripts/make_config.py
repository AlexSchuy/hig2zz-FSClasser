import argparse
from configparser import ConfigParser

from common import main_setup, utils
from data import process


def main():
    main_setup.setup()
    parser = argparse.ArgumentParser(
        description='Make a config file for training.')
    parser.add_argument('--processes', nargs='*',
                        help='List of processes to include.')
    parser.add_argument('--output', required=True,
                        help='The name of the output config file.')
    parser.add_argument('--features', nargs='+',
                        help='List of features to include.')
    parser.add_argument('--version', '-v', required=True,
                        help='The CEPC version to use.')
    args = parser.parse_args()
    if not args.processes:
        args.processes = process.get_background_processes(args.version)
    config = ConfigParser()
    config['DATA'] = {}
    config['DATA']['feature_keys'] = utils.list_to_str(args.features)
    config['DATA']['processes'] = utils.list_to_str(args.processes)
    config['DATA']['version'] = args.version
    with open(args.output, 'w') as config_file:
        config.write(config_file)


if __name__ == '__main__':
    main()
