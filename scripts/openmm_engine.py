#!/usr/bin/env python
from openmmtools.alchemy import *
from openmm_readinput import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from time import time

# TODO: Add support for CHARMM Drude polarizable FF
# TODO: Add an attractive Morse term so that the droplet density is 0.033 mol/A^3
# TODO: Add WCA Perturbation Option in Alchemical Region
# TODO: Add option for PBC system

# User Input
inp = read_config(sys.argv[1])
inp.initialize()

# Hardware
platform = Platform.getPlatformByName(inp.Hardware_type)
if inp.Hardware_type == 'CUDA':
	properties = {'CudaDeviceIndex': inp.Hardware_gpu_idx, 
                  'CudaPrecision'  : inp.Hardware_gpu_prec}

# Load CHARMM files
psf    = read_psf(inp.structure)
pdb    = read_pdb(inp.coordinates)
params = read_params(inp.param_file)

# Create System Object
system = psf.createSystem(params,
                          nonbondedMethod=NoCutoff,
                          nonbondedCutoff=inp.Droplet_cutoff*angstrom,
                          constraints=inp.constraint,
                          rigidWater=True)

# Spherical Boundary Condition
r0 = inp.Droplet_radius - (inp.kBT/inp.Droplet_k)**0.5
print("Shifted Radius: %.2f" % r0)
spherical_boundary = CustomExternalForce('(ks/2.0)*max(0, r-r0)^2;'
                                         'r=sqrt(x*x + y*y + z*z)')
spherical_boundary.addGlobalParameter('ks', inp.Droplet_k*kilocalories_per_mole/angstrom**2)
spherical_boundary.addGlobalParameter('r0', r0*angstrom)

# Solute center of charge
center_of_charge = CustomExternalForce('(kr/2.0)*(x*x + y*y + z*z)')
center_of_charge.addGlobalParameter('kr', inp.Solute_k*kilocalories_per_mole/angstrom**2)

# Atom selections
I = 0
for line in open(inp.coordinates):
	if line.startswith('ATOM') or line.startswith('HETATM'):
		dummy = line.split()
		if dummy[2] == inp.Solute_center_atom:
			center_of_charge.addParticle(I, [])
		if dummy[2] == inp.Droplet_atom:
			spherical_boundary.addParticle(I, [])
		I += 1

# Push force object to system
system.addForce(spherical_boundary)
system.addForce(center_of_charge)

# Langevin Dynamics
integrator = LangevinIntegrator(inp.temperature*kelvin,
                                inp.fric_coeff/picosecond,
                                inp.dt*femtoseconds)

# Transform system for Alchemical calculation
if inp.Run_type == 'FEP':
	factory = AbsoluteAlchemicalFactory(consistent_exceptions=False,
                                            disable_alchemical_dispersion_correction=True)
	alchemical_region = AlchemicalRegion(alchemical_atoms=inp.FEP_atoms,
                                             annihilate_electrostatics=inp.FEP_anni_elec,
                                             annihilate_sterics=inp.FEP_anni_vdw)
	system = factory.create_alchemical_system(system,
                                              alchemical_region)

# Simulation Parameters
simulation = Simulation(psf.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)

# MD Equilibration
if inp.Run_type == 'MD':
	# Minimize System
	if inp.min_nstep > 0:
		print("Minimizing Energy")
		simulation.minimizeEnergy(tolerance=inp.min_tol*kilocalories_per_mole)
		simulation.context.setVelocitiesToTemperature(inp.temperature*kelvin)
	
	# Reporter
	if inp.reporter == 'on':
		simulation.reporters.append(StateDataReporter(
                                   'MD_equil.log',
                                   inp.nstout,
                                   step=True,
                                   kineticEnergy=True,
                                   potentialEnergy=True,
                                   totalEnergy=True,
                                   temperature=True,
                                   totalSteps=inp.MD_total,
                                   progress=True,
                                   remainingTime=True,
                                   speed=True,
                                   separator=" "))
	
	# Equilibration
	if inp.MD_equil > 0:
		print("Running Equilibration...")
		start = time()
		simulation.step(inp.MD_equil)
		end = time()
		print("Walltime: %.2f sec" % (end-start))
		
	# Production
	if inp.dcdreporter == 'on':
		simulation.reporters.append(DCDReporter('output-equil.dcd', inp.nstdcd))
	print("Running Production...")
	start = time()
	simulation.step(inp.MD_production)
	end = time()
	print("Walltime: %.2f sec" % (end-start))
	if inp.output_pdb:
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open(inp.output_pdb, 'w'))

if inp.Run_type == 'FEP':
	if inp.reporter == 'on':
		simulation.reporters.append(StateDataReporter(
                                   'FEP_%s.log' % inp.FEP_type,
                                   inp.nstout,
                                   step=True,
                                   kineticEnergy=True,
                                   potentialEnergy=True,
                                   totalEnergy=True,
                                   temperature=True,
                                   totalSteps=inp.FEP_total,
                                   progress=True,
                                   remainingTime=True,
                                   speed=True,
                                   separator=" "))
	if inp.dcdreporter == 'on':
		simulation.reporters.append(DCDReporter('output-%s.dcd' % inp.FEP_type, inp.nstdcd))
	
	# Initialzie Perturbed Hamiltonian through AlchemicalState
	alchemical_state = AlchemicalState.from_system(system)
	alchemical_state.lambda_sterics = 1.0
	if inp.FEP_type == 'vdw':
		alchemical_state.lambda_electrostatics = 0.0
	elif inp.FEP_type == 'elec':
		alchemical_state.lambda_electrostatics = 1.0
	alchemical_state.apply_to_context(simulation.context)

	# Alchemical Simulation
	start = time()
	for k in range(inp.FEP_nstates):
		# Set Lambda value
		print('Setting Lambda Value: %f' % inp.FEP_lambda[k])
		if inp.FEP_type == 'vdw':
			alchemical_state.lambda_sterics = inp.FEP_lambda[k]
		elif inp.FEP_type == 'elec':
			alchemical_state.lambda_electrostatics = inp.FEP_lambda[k]
		alchemical_state.apply_to_context(simulation.context)
	
		# Equilibration Run
		t11 = time()
		simulation.step(inp.FEP_equil)
		t12 = time()
		print('\tEquilibration Time: \t %.2f sec' % (t12-t11))

		# Production Run
		t21 = time()
		for iteration in range(inp.FEP_niter):
			if inp.FEP_type == 'vdw':
				alchemical_state.lambda_sterics = inp.FEP_lambda[k]
			elif inp.FEP_type == 'elec':
				alchemical_state.lambda_electrostatics = inp.FEP_lambda[k]
			alchemical_state.apply_to_context(simulation.context)
		
			# run Dynamics
			simulation.step(inp.FEP_isteps)
		
			# Compute energies at all alchemical states and store
			for l in range(inp.FEP_nstates):
				if inp.FEP_type == 'vdw':
					alchemical_state.lambda_sterics = inp.FEP_lambda[l]
				elif inp.FEP_type == 'elec':
					alchemical_state.lambda_electrostatics = inp.FEP_lambda[l]
				alchemical_state.apply_to_context(simulation.context)
				inp.FEP_u_kln[k,l,iteration] = simulation.context.getState(getEnergy=True).getPotentialEnergy() / inp.kT
		t22 = time()
		print('\tProduction Time   : \t %.2f sec' % (t22-t21))
		print('\tTotal Time Taken  : \t %.2f sec' % (t22-t11))

	end = time()
	
	# Save reduced potential
	np.save(inp.FEP_pot_name, inp.FEP_u_kln)
	if inp.output_pdb:
		positions = simulation.context.getState(getPositions=True).getPositions()
		PDBFile.writeFile(simulation.topology, positions, open(inp.output_pdb, 'w'))
	
	print('=========================================================================')
	print('Total time taken: %f sec' % (end-start))
	print('=========================================================================')

