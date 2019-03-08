#!/usr/bin/env bash

unset MARLIN_DLL
#source /besfs/groups/higgs/Software/v01-17-05_slc6/init_ilcsoft_151105.sh
source /besfs/groups/higgs/Software/v01-17-05_slc6/init_ilcsoft.sh

# Add FSClasser to the shared library path
export FSClasser_HOME=/workfs/atlas/guofy/cepc/FSClasser
export LD_LIBRARY_PATH="$FSClasser_HOME/lib:$LD_LIBRARY_PATH"
export MARLIN_DLL=$MARLIN_DLL:$FSClasser_HOME/lib/libFSClasser.so

# Add Higgs2zz to the shared library path
export LD_LIBRARY_PATH=$PWD/Higgs2zz/lib:$LD_LIBRARY_PATH
export MARLIN_DLL=$MARLIN_DLL:$PWD/Higgs2zz/lib/libHiggs2zz.so
