unset MARLIN_DLL
source /cvmfs/cepc.ihep.ac.cn/software/cepcenv/setup.sh
cepcenv use 0.1.0-rc9

# Add FSClasser to the shared library path
export FSClasser_HOME=/workfs/bes/lig/higgs/FSClasser
export LD_LIBRARY_PATH="$FSClasser_HOME/lib:$LD_LIBRARY_PATH"
export MARLIN_DLL=$MARLIN_DLL:$FSClasser_HOME/lib/libFSClasser.so

# Add Higgs2zz to the shared library path
export LD_LIBRARY_PATH=$PWD/Higgs2zz/lib:$LD_LIBRARY_PATH
export MARLIN_DLL=$MARLIN_DLL:$PWD/Higgs2zz/lib/libHiggs2zz.so

