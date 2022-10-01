import json
import h5py
import numpy as np

# from neurogenome.genome import Genome
from genome import Genome


def save_genome(path: str, genome: Genome) -> None:
	""" 
	Метод сохранения генома
	**************
	
	Parameters
	----------
	path : str
		относительный путь
	genome : Genome
		сохраняемый геном
	"""
	if ".hdf5" in path:
		with h5py.File(path, 'w') as f:
			f.attrs["settings"] = json.dumps(genome.settings)
			_ = f.create_dataset("promoters", data=genome.genes[0])
			_ = f.create_dataset("kernels", data=genome.genes[1])
	# elif ".npy" in path:
	# 	with open(path, 'wb') as f:
	# 		np.save(f, genome.genes[0])
	# 		np.save(f, genome.genes[1])
	else:
		raise Exception("Неподдерживаемый тип файла")

def load_genome(path: str) -> Genome:
	"""
	Метод загрузки генома
	**************
	
	Parameters
	----------
	path : str
		относительный путь
	"""
	if ".hdf5" in path:
		settings = dict()
		with h5py.File(path, 'r') as f:
			settings = json.loads(f.attrs["settings"])
			promoters = np.array(f["promoters"])
			kernels = np.array(f["kernels"])
	# elif ".npy" in path:
	# 	with open(path, 'rb') as f:
	# 		promoters = np.load(f, allow_pickle=True)
	# 		kernels = np.load(f, allow_pickle=True)
	else:
		raise Exception("Неподдерживаемый тип файла")

	return Genome(
		settings,
		[promoters, kernels]
	)