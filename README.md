# Info
* OpenMM python scripts to calculate solvation free energies of ions and charged molecules using spherical boundary conditions (SBC)
* The ion/molecule is alchemically decoupled using the alchemy package as part of openmmtools and the free energy is estimated with MBAR.
* The self-energy for molecules is calculated using a generalized Born equation derived in the paper below. The python script uses MDAnalysis to read in the trajectory file.

Setiadi, J. and Kuyucak, S., 2019. A simple, parameter-free method for computing solvation free energies of ions. The Journal of Chemical Physics, 150(6), p.065101.
