#!/usr/bin/env python
from pymbar import MBAR, timeseries
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Command line arguments
parser = argparse.ArgumentParser(
description="""Calculates free energy with MBAR from simulation output. The script reads 
in the reduced potential written as a numpy binary file *.npy"""
)
parser.add_argument('-pot', dest='pot',  help='Numpy matrix file for reduced potential', required=True)
parser.add_argument('-fep', dest='fep',  help='FEP type, elec to vdw (naming purposes)', required=True)
parser.add_argument('-out', dest='out',  help='Output file (default: mbar-fep.txt)', default='fep')
parser.add_argument('-T',   dest='temp', help='Temperature', required=True)
args = parser.parse_args()

Filename = args.pot
Savename = 'mbar-' + args.fep + '.txt'
FEP      = args.fep
Temp     = float(args.temp)
kB       = 0.001987204118
kT       = kB*Temp

u_kln = np.load(Filename)
nstates, m, k = np.shape(u_kln)
l = np.linspace(0,1,nstates)

# Subsample data to extract uncorrelated equilibrium timeseries
N_k = np.zeros([nstates], np.int32) # number of uncorrelated samples
for k in range(nstates):
    [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k,k,:])
    indices = timeseries.subsampleCorrelatedData(u_kln[k,k,:], g=g)
    N_k[k] = len(indices)
    u_kln[k,:,0:N_k[k]] = u_kln[k,:,indices].T

# Compute free energy differences and statistical uncertainties
mbar = MBAR(u_kln, N_k)
[DeltaF_ij, dDeltaF_ij, Theta_ij] = mbar.getFreeEnergyDifferences()
ODeltaF_ij = mbar.computeOverlap()

# Print results
f = open(Savename, 'w')
for i in range(nstates):
	f.writelines("%.2f: %9.4f +- %.4f\n" % (l[i], DeltaF_ij[i,0]*kT, dDeltaF_ij[i,0]*kT))
f.close()

# Plot Overlap
fig1, ax1 = plt.subplots()
plt.imshow(ODeltaF_ij[2])
plt.set_cmap('bone')
cbar = plt.colorbar()
cbar.set_label("Overlap",fontsize=12)
ax1.xaxis.tick_top()
plt.clim(0,1)
plt.title("Lambda", pad=24,fontsize=14)
plt.ylabel("Lambda",fontsize=14)
fig1.savefig("MBAR-Overlap-%s.png" % FEP, dpi=fig1.dpi, bbox_inches='tight')

# Plot Energies
fig2 = plt.figure()
plt.errorbar(l,DeltaF_ij[:, 0]*kT,yerr=dDeltaF_ij[:,0]*kT, marker='o', ms=2, mew=4)
plt.xlabel("lambda",fontsize=14)
plt.ylabel("ddG (kcal/mol)",fontsize=14)
fig2.savefig("MBAR-FEP-%s.png" % FEP, dpi=fig2.dpi, bbox_inches='tight')

