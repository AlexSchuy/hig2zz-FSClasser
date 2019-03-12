# hig2zz_FSClasser
Analysis of H -> ZZ channels at CEPC using FSClasser for final state selection.

## Dependencies
This project relies upon Python 3.6 with pyROOT bindings (note: this means you need to compile ROOT against Python 3.6, rather than the
usual Python 2.7). To manage Python packages, pipenv is used (see https://pipenv.readthedocs.io/en/latest/ for more information).

## Setup
Source `scripts/install.sh` and `scripts/setup.sh` to properly setup your environment. `scripts/install.sh` needs to be sourced only once,
but `scripts/setup.sh` should be sourced every session. There are several config files in the config directory managing sample locations, 
cross-sections, etc., but probably the only one you need to edit is `settings.ini.example`. To do so, first run `cp config/settings.ini.example config/settings.ini`
and then edit the entries in `config/settings.ini` that are preceded by '# EDIT'. Finally, to enter the virtualenv, run `pipenv shell`.

## Preface
Note that many of the files in this repository contain a `main()` function which can be run. Please add the `-h` arg when running a file
to see a list of possible arguments and an explanation of the purpose of the file.

## Steps
This will guide you through the steps necessary to receive output ntuples that can be used for statistical analysis:

1. Create final state information. To do so, run `python final_state_selection/job_runner.py` (remember to pass `-h` to see a description of the inputs). Then, run `python final_state_selection/validate_jobs.py` to ensure the jobs ran correctly.
2. Modify the fs data for event selection. To do so, run the 'preprocess' and 'merge' subparsers of `data/setup_data.py` (for example, `python data/setup_data.py preprocess -h` to see arguments for the preprocess subparser)
3. Train the ML model by running `python training/train.py`. train.py will expect a config file listing the processes, CEPC version, variables, etc. that you wish to use. Some configs are already included in the `training` directory, or you can write your own either manually or with `scripts/make_config.py`. Whenever you run `train.py`, it creates a new 'run directory' that will store the results of training and also output ntuples and plots. `train.py` will inform you of the location of the run dir it is using. The default location for run dirs is listed in `config/settings.ini` as 'run_dir' under '[Event Selection]'.
4. If you wish to make plots of the final data, run `python plotting/metrics.py`. Otherwise, to obtain final root ntuples, run `python scripts/to_root.py`. Both files take the name of a run dir as input (for example, '000', '001', etc.), but they will use the most-recent run dir if you do not specify one. The ntuples will be stored in the 'ntuples' directory within the run dir.
