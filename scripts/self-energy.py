#!/usr/bin/env python
from numpy.linalg import norm
import numpy as np
import MDAnalysis
import argparse

# Command line arguments
parser = argparse.ArgumentParser(
description="Calculates the self-energy of a charged molecule inside a spherical droplet.",
epilog='"The cure for anything is salt water-tears, sweat or the sea. - Isak Dinesen"'
)
parser.add_argument('-psf', dest='psf', help='Input CHARMM PSF file', required=True)
parser.add_argument('-dcd', dest='dcd', help='Input trajectory DCD file', required=True)
parser.add_argument('-out', dest='out', help='Output file name (default: self-energy.txt)', default='self-energy.txt')
parser.add_argument('-sel', dest='sel', help='Resname of Molecule', required=True)
parser.add_argument('-rad', dest='rad', help='Radius of spherical droplet (Ang)', required=True)
parser.add_argument('-eps', dest='eps', help='Dielectric constant (default: 80.0)', default=80.0)
args = parser.parse_args()

# Load parameters
PSF      = args.psf
DCD      = args.dcd
OUT      = args.out
Sel      = "resname " + args.sel
Rad      = float(args.rad)
eps      = float(args.eps)

# Initialize
universe = MDAnalysis.Universe(PSF, DCD)
molecule = universe.select_atoms(Sel)
qcharges = molecule.atoms.charges
qtotal   = sum(qcharges)
Niframes = len(universe.trajectory)
Ncharges = len(molecule)

Cselfs   = np.zeros(Niframes, dtype=np.float64)
Ccross   = np.zeros(Niframes, dtype=np.float64)
t        = np.linspace(1, Niframes, Niframes)
conv     = 331.31973577619925

# Loop over frames
I = 0
for ts in universe.trajectory:
    # Self-Interaction
    Rposi = norm(molecule.positions, axis=1)
    Rself = Rad**2 - Rposi**2
    Cselfs[I] = sum(Rad * qcharges*qcharges / Rself)

    # Cross-Interaction
    for i in range(Ncharges - 1):
        v1 = molecule.positions[i]
        r1 = norm(v1)
        for j in range(i+1, Ncharges):
            v2 = molecule.positions[j]
            r2 = norm(v2)
            dr = (Rad**4 + (r1*r2)**2 - 2*Rad**2*np.dot(v1,v2))**0.5
            Ccross[I] += Rad*qcharges[i]*qcharges[j]/dr

    I += 1

# Self-energy
Uself = -conv/2.0*(1.0 - 1.0/eps) * (Cselfs + 2*Ccross)
UBorn = -conv*qtotal**2 / (2.0*Rad) * (1.0 - 1.0/eps)
A     = -conv/2.0*(1.0 - 1.0/eps) * Cselfs
B     = -conv/2.0*(1.0 - 1.0/eps) * 2*Ccross

# Save Results
f = open(OUT, "w")
f.writelines("# Molecule             : %s\n" % Sel)
f.writelines("# Total Charge         : %.1fe\n" % qtotal)
f.writelines("# Droplet Radius       : %.1f Ang\n" % Rad)
f.writelines("# Dielectric Const     : %.1f\n" % eps)
f.writelines("#\n")
f.writelines("# Born Equation        : %.4f kcal/mol\n" % UBorn)
f.writelines("# Generalized Equation : %.4f +- %.4f kcal/mol\n" % (np.mean(Uself), np.std(Uself)))
f.writelines("#\n")
f.writelines("#%9s %10s %10s %10s\n" % ("Frame", "Energy", "Diagonal", "Cross"))
f.writelines("#%9s %10s %10s %10s\n" % ("-----", "------", "--------", "-----"))
for i in range(Niframes):
    f.writelines("%10i %10.4f %10.4f %10.4f\n" % (t[i], Uself[i], A[i], B[i]))
f.close()
