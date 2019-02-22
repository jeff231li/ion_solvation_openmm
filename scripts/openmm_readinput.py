from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np

class _OpenMMSBCConfig():
	def __init__(self):
		self.nstout             = 500
		self.nstdcd             = 500
		self.reporter           = 'on'
		self.dcdreporter        = 'off'
		self.output_pdb         = None
		
		self.dt                 = 2.0
		self.fric_coeff         = 1.0
		self.temperature        = 298.0
		self.constraint         = HBonds
		
		self.Droplet_atom       = 'OH2'
		self.Droplet_cutoff     = 100.0
		
		self.Solute_k           = 1.0
		
		self.Run_type           = 'MD'
		
		self.min_nstep          = 0
		self.min_tol            = 0.1
		
		self.MD_equil           = 0
		self.MD_production      = 50000
		
		self.FEP_type           = 'elec'
		self.FEP_nlambda        = None
		self.FEP_lambda         = None
		self.FEP_lambda_dir     = 1
		self.FEP_equil          = 50000
		self.FEP_production     = 50000
		self.FEP_niter          = 500
		self.FEP_pot_name       = 'potential.npy'
		
		self.FEP_anni_elec      = False
		self.FEP_anni_vdw       = False
		
		self.Hardware_type      = 'CUDA'
		self.Hardware_gpu_prec  = 'mixed'
		self.Hardware_gpu_idx   = '0'
		
	def initialize(self):
		self.kBT      = 0.001987204118*self.temperature
		self.kT       = AVOGADRO_CONSTANT_NA*BOLTZMANN_CONSTANT_kB*self.temperature*kelvin
		self.MD_total = self.MD_equil + self.MD_production
		if self.Run_type == 'FEP':
			if self.FEP_nlambda and not self.FEP_lambda:
				self.FEP_lambda = np.linspace(0, 1, self.FEP_nlambda)
			if self.FEP_lambda_dir == -1:
				self.FEP_lambda = self.FEP_lambda[::-1]
			self.FEP_nstates = len(self.FEP_lambda)
			self.FEP_niter   = int(self.FEP_production/self.FEP_isteps)
			self.FEP_total   = (self.FEP_equil + self.FEP_production)*self.FEP_nstates
			self.FEP_u_kln   = np.zeros([self.FEP_nstates,self.FEP_nstates,self.FEP_niter], np.float64)
		
	def readconfig(self, inputFile):
		for line in open(inputFile, 'r'):
			if not line.startswith("#"):
				if line.find("#") >= 0:
					line = line.split("#")[0]
				dummy = line.strip().split("=")
				if len(dummy) > 1:
					inp_param = dummy[0].upper().strip()
					try:
						inp_value = dummy[1].strip()
					except:
						inp_value = None
					if inp_value:
						if inp_param == 'STRUCTURE':
							self.structure = inp_value
						if inp_param == 'COORDINATES':
							self.coordinates = inp_value
						if inp_param == 'OUTPUT_PDB':
							self.output_pdb = inp_value
						if inp_param == 'PARAM_FILE':
							self.param_file = inp_value
						if inp_param == 'NSTOUT':
							self.nstout = int(inp_value)
						if inp_param == 'NSTDCD':
							self.nstdcd = int(inp_value)
						if inp_param == 'REPORTER':
							self.reporter = inp_value
						if inp_param == 'DCDREPORTER':
							self.dcdreporter = inp_value
						if inp_param == 'TIME_STEP':
							self.dt = float(inp_value)
						if inp_param == 'FRIC_COEFF':
							self.fric_coeff = float(inp_value)
						if inp_param == 'TEMPERATURE':
							self.temperature = float(inp_value)
						if inp_param == 'CONSTRAINT':
							if inp_value == 'NONE':
								self.constraint = None
							if inp_value == 'HBONDS':
								self.constraint = HBonds
							if inp_value == 'ALLBONDS':
								self.constraint = AllBonds
							if inp_value == 'HANGLES':
								self.constraint = HAngles
						if inp_param == 'DROPLET_RADIUS':
							self.Droplet_radius = float(inp_value)
						if inp_param == 'DROPLET_ATOM':
							self.Droplet_atom = inp_value
						if inp_param == 'DROPLET_K':
							self.Droplet_k = float(inp_value)
						if inp_param == 'DROPLET_CUTOFF':
							self.Droplet_cutoff = float(inp_value)
						if inp_param == 'SOLUTE_CENTER_ATOM':
							self.Solute_center_atom = inp_value
						if inp_param == 'SOLUTE_K':
							self.Solute_k = float(inp_value)
						if inp_param == 'RUN_TYPE':
							self.Run_type = inp_value
						if inp_param == 'MIN_NSTEPS':
							self.min_nstep = int(inp_value)
						if inp_param == 'MIN_TOL':
							self.min_tol = float(inp_value)
						if inp_param == 'MD_EQUIL':
							self.MD_equil = int(inp_value)
						if inp_param == 'MD_PRODUCTION':
							self.MD_production = int(inp_value)
						if inp_param == 'FEP_TYPE':
							self.FEP_type = inp_value
						if inp_param == 'FEP_ATOM_IDX':
							temporary = inp_value.split('-')
							ith = int(temporary[0])
							if len(temporary) == 1:
								self.FEP_atoms = [ith]
							else:
								jth = int(temporary[1])
								self.FEP_atoms = [i for i in range(ith, jth+1)]
						if inp_param == 'FEP_EQUIL':
							self.FEP_equil = int(inp_value)
						if inp_param == 'FEP_PRODUCTION':
							self.FEP_production = int(inp_value)
						if inp_param == 'FEP_NLAMBDA':
							self.FEP_nlambda = int(inp_value)
						if inp_param == 'FEP_LAMBDA':
							temporary = dummy[-1].split()
							self.FEP_lambda = [float(temporary[i]) for i in range(len(temporary))]
						if inp_param == 'FEP_LAMBDA_DIR':
							self.FEP_lambda_dir = int(inp_value)
						if inp_param == 'FEP_ISTEPS':
							self.FEP_isteps = int(inp_value)
						if inp_param == 'FEP_POT_NAME':
							self.FEP_pot_name = inp_value
						if inp_param == 'FEP_ANNI_ELEC':
							if inp_value == 'yes':
								self.FEP_anni_elec = True
							elif inp_value == 'no':
								self.FEP_anni_elec = False
						if inp_param == 'FEP_ANNI_VDW':
							if inp_value == 'yes':
								self.FEP_anni_vdw = True
							elif inp_value == 'no':
								self.FEP_anni_vdw = False
						if inp_param == 'HARDWARE_TYPE':
							self.Hardware_type = inp_value
						if inp_param == 'HARDWARE_GPU_PREC':
							self.Hardware_gpu_pre = inp_value
						if inp_param == 'HARDWARE_GPU_IDX':
							self.Hardware_gpu_idx = inp_value
		return self

def read_config(configFile):
	return _OpenMMSBCConfig().readconfig(configFile)

def read_psf(filename):
	return CharmmPsfFile(filename)

def read_pdb(filename):
	return PDBFile(filename)

def read_params(filename):
	charmmExt = ['rtf', 'prm', 'str']
	paramFiles = ()
	
	for line in open(filename, 'r'):
		if line.find("#") >= 0:
			line = line.split("#")[0]
		parfile = line.strip()
		if len(parfile) > 0:
			ext = parfile.lower().split('.')[-1]
			if ext in charmmExt:
				paramFiles += ( parfile, )
	
	return CharmmParameterSet( *paramFiles )

