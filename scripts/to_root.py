import os
from array import array

from root_numpy import array2root

from common import utils
from config import settings
from data import load, process

feature_name_map = {'RMass1234': 'Mass_invar'}
RUN_DIR = settings.get_setting('Event Selection', 'run_dir')


def write_root(run_dir, filename, X, weights):
    utils.makedirs(os.path.join(run_dir, 'ntuple'))
    filename = os.path.join(run_dir, 'ntuple', filename)
    X['weights'] = weights
    array2root(X.to_records(index=False), filename, 'HiggsTree', 'recreate')


def to_root(run_dir):
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

    # Acquire ntuples from selected events.
    selected_indices = model.predict(X_test) == 1
    y_selected = y_test[selected_indices]
    X_selected = X_test.loc[selected_indices]

    # Write appropriate variables (depending on the signal process) for each event
    # split into several root files (signal, SM background, and each ZH background
    # ).
    signal_process = process.get_signal_process(version, dataset)
    if signal_process == 'nnh_zz':
        output_features = ['RMass1234']
    else:
        raise NotImplementedError(
            '{} analysis is not supported yet.'.format(signal_process))
    X_selected = X_selected[output_features]
    X_selected, weights = process.by_group(version, dataset, process.get_signal_sm_zh_background_groups(
        version, dataset), y_selected, [X_selected], weight_type='expected')
    write_root(run_dir=run_dir, filename='{}_sig.root'.format(
        signal_process), X=X_selected['signal'], weights=weights['signal'])
    write_root(run_dir=run_dir, filename='{}_sm_bkg.root'.format(
        signal_process), X=X_selected['sm_background'], weights=weights['sm_background'])
    write_root(run_dir=run_dir, filename='{}_zh_bkg.root'.format(
        signal_process), X=X_selected['zh_background'], weights=weights['zh_background'])


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Write ntuples from events that passed analysis selection to root files for cross section X branching ratio and higgs width meaasurements.')
    parser.add_argument('--run_dir', default=None, help='The run dir to use.')

    if args.run_dir:
        run_dir = os.path.join(RUN_DIR, args.run_dir)
    else:
        run_dir = utils.most_recent_dir()


if __name__ == '__main__':
    main()
