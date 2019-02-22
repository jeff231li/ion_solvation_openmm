#!/bin/bash

# Export CUDA and Anaconda
export PATH='/home/peanut/anaconda3/bin:$PATH'
export PATH=/usr/local/cuda-9.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/openmpi/lib/:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=/usr/local/cuda-9.0/lib:${DYLD_LIBRARY_PATH}

# Run equilibration
python ../../scripts/openmm_engine.py omm-equil.inp

# Run FEP
#python ../../scripts/openmm_engine.py omm-elec.inp
#python ../../scripts/openmm_engine.py omm-vdw.inp

# Analysis
python ../../scripts/self-energy.py -psf sodium.psf -dcd output-equil.dcd -rad 24.0 -sel SOD
#python ../../scripts/mbar-fep.py -pot reduced-elec.npy -fep elec -T 300
#python ../../scripts/mbar-fep.py -pot reduced-vdw.npy -fep vdw -T 300

exit
