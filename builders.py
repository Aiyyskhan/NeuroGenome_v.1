from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# from neurogenome.genome import Genome, I_DTYPE, F_DTYPE
from genome import Genome #, I_DTYPE #, F_DTYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

F_DTYPE = torch.float32
# I_DTYPE = torch.int32

SETTINGS = {
	"population size": 50,
	"number of leaders": 5,
	"select by": "max",
	"matrix shape": [[3,7],[3,8],[5,15],[120,110]], # form must be specified as a list, not as a tuple
}

def template_maker(promoters, pop_size, n_t_rows, n_t_cols):
	templates = torch.zeros(size=(pop_size, 9 * n_t_rows, 9 * n_t_cols), dtype=F_DTYPE, device=DEVICE)
	templates[:, :, np.arange(4, templates.shape[2], 9)] += 1.0
	templates[:, np.arange(4, templates.shape[1], 9), :] += 1.0
	templates[templates < 2] = 0.0
	templates[templates == 2] = promoters.ravel()
	return templates

def nca_func(templates, pop_size, kernels):
	"""
	Neural cellular automaton

	**************
	
	Parameters
	----------
	"""
	# padding = 1 # for kernel shape (3,3)
	padding = 2 # for kernel shape (5,5)

	matrices = templates[None,:,:,:]

	p0 = F.conv2d(matrices, kernels[:, 0], padding=padding, groups=pop_size)
	p1 = F.conv2d(matrices, kernels[:, 1], padding=padding, groups=pop_size)

	a = F.conv2d(matrices, kernels[:, 2], padding=padding, groups=pop_size)
	b = F.conv2d(p0, kernels[:, 3], padding=padding, groups=pop_size)
	c = F.conv2d(p1, kernels[:, 4], padding=padding, groups=pop_size)
	# p = torch.atan(a + b + c)
	p = a + b + c

	# u = torch.relu(F.conv2d(p, kernels[:, 5], padding=padding, groups=pop_size))
	# u = torch.sigmoid(F.conv2d(p, kernels[:, 5], padding=padding, groups=pop_size))
	u = torch.atan(F.conv2d(p, kernels[:, 5], padding=padding, groups=pop_size))
	# u = torch.relu(torch.atan(F.conv2d(p, kernels[:, 5], padding=padding, groups=pop_size)))
	# u = torch.tanh(F.conv2d(u, kernels[:, 6], padding=padding, groups=pop_size))
	u = F.conv2d(u, kernels[:, 6], padding=padding, groups=pop_size)
	# u = torch.atan(F.conv2d(u, kernels[:, 6], padding=padding, groups=pop_size))

	m = torch.relu(F.conv2d(p, kernels[:, 7], padding=padding, groups=pop_size))
	# # m = torch.relu(torch.atan(F.conv2d(p, kernels[:, 7], padding=padding, groups=pop_size)))
	# m = torch.atan(F.conv2d(p, kernels[:, 7], padding=padding, groups=pop_size))
	# # m = torch.sigmoid(F.conv2d(p, kernels[:, 7], padding=padding, groups=pop_size))
	m = torch.sigmoid(F.conv2d(m, kernels[:, 8], padding=padding, groups=pop_size))
	# m = torch.atan(F.conv2d(m, kernels[:, 8], padding=padding, groups=pop_size))

	# r = torch.relu(F.conv2d(p, kernels[:, 9], padding=padding, groups=pop_size))
	# # r = torch.relu(torch.atan(F.conv2d(p, kernels[:, 9], padding=padding, groups=pop_size)))
	# # r = torch.atan(F.conv2d(p, kernels[:, 9], padding=padding, groups=pop_size))
	# r = torch.sigmoid(F.conv2d(r, kernels[:, 10], padding=padding, groups=pop_size))

	# matrices = matrices * (1.0 - m) + u * m
	# matrices = matrices + u * m
	matrices = u * m

	matrices[matrices.absolute() < 0.1] = 0.0

	return matrices.cpu().numpy()

def nca_func_1(templates, pop_size, kernels):
	"""
	Neural cellular automaton

	**************
	
	Parameters
	----------
	"""
	matrices = templates[None,:,:,:]

	p0 = F.conv2d(matrices, kernels[:, 0], padding=1, groups=pop_size)
	p1 = F.conv2d(matrices, kernels[:, 1], padding=1, groups=pop_size)

	moore = torch.broadcast_to(torch.tensor(np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]),\
			dtype=F_DTYPE, device=DEVICE), (pop_size,1,3,3)) #.reshape((pop_size,1,3,3))
	sobel_y = torch.broadcast_to(torch.tensor(np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]),\
			dtype=F_DTYPE, device=DEVICE), (pop_size,1,3,3))
	sobel_x = torch.broadcast_to(torch.tensor(np.array([[[-1, 2, -1], [0, 0, 0], [1, 2, 1]]]), \
			dtype=F_DTYPE, device=DEVICE), (pop_size,1,3,3))

	moore = moore / torch.tensor([8],dtype=F_DTYPE, device=DEVICE)

	a = F.conv2d(matrices, sobel_x, padding=1, groups=pop_size)
	b = F.conv2d(p0, sobel_y, padding=1, groups=pop_size)
	c = F.conv2d(p1, moore, padding=1, groups=pop_size)
	p = a + b + c

	u = torch.sigmoid(F.conv2d(p, kernels[:, 2], padding=1, groups=pop_size))
	u = F.conv2d(u, kernels[:, 3], padding=1, groups=pop_size)
	# u = torch.atan(F.conv2d(u, kernels[:, 3], padding=1, groups=pop_size))

	m = torch.relu(F.conv2d(p, kernels[:, 4], padding=1, groups=pop_size))
	m = torch.sigmoid(F.conv2d(m, kernels[:, 5], padding=1, groups=pop_size))
	# m = torch.atan(F.conv2d(m, kernels[:, 5], padding=1, groups=pop_size))

	# matrices = matrices * (1.0 - m) + u * m
	matrices = matrices + u * m

	# matrices[(matrices > -0.1) & (matrices < 0.1)] = 0.0
	matrices[matrices.absolute() < 0.1] = 0.0

	return matrices.cpu().numpy()

def genome_builder(settings: Dict[str, Any]) -> Genome:
	"""
	Метод-сборщик генома

	**************
	
	Parameters
	----------
	settings : Dict[str, Any]
		словарь с основными настройками (гиперпараметрами)
	"""

	# for promoter
	shape_arr = np.array(settings["matrix shape"])
	num_tiles = np.ceil((shape_arr[:,0] * shape_arr[:,1]) / 9.0).sum().astype(np.int32)
	promoters_shape = (settings["population size"], num_tiles)
	# promoters = np.random.random(size=promoters_shape)
	# promoters = np.random.randn(*promoters_shape)
	# promoters = np.random.randint(-127, 128, size=promoters_shape)
	promoters = np.random.randint(-256, 257, size=promoters_shape)
	# promoters = np.random.randint(-64, 65, size=promoters_shape)

	# for nca_func
	nca_kernels_shape = (settings["population size"], 9,5,5)
	# nca_kernels_shape = (settings["population size"], 9,3,3)

	# nca_kernels = np.random.random(size=nca_kernels_shape)
	# nca_kernels = np.random.randn(*nca_kernels_shape)
	# nca_kernels = np.random.randint(-127, 128, size=nca_kernels_shape)
	nca_kernels = np.random.randint(-256, 257, size=nca_kernels_shape)
	# nca_kernels = np.random.randint(-64, 65, size=nca_kernels_shape)

	genes = [promoters, nca_kernels]
	
	return Genome(
		settings,
		genes
	)

def neuro_builder(genome: Genome) -> List:
	"""
	Метод-сборщик нейросетевых матриц из генома

	**************
	
	Parameters
	----------
	genome : Genome
		геном
	"""
	# pop_size = genome.settings["population size"]
	pop_size = genome.num_individuals
	promoters = torch.tensor(genome.genes[0] / 128.0).to(dtype=F_DTYPE, device=DEVICE)
	nca_kernels = torch.tensor(genome.genes[1] / 128.0).to(dtype=F_DTYPE, device=DEVICE).reshape((pop_size,9,1,5,5))
	# nca_kernels = torch.tensor(genome.genes[1] / 128.0).to(dtype=F_DTYPE, device=DEVICE).reshape((pop_size,9,1,3,3))

	lay_list = []
	tile_count = 0
	for lay_shape in genome.settings["matrix shape"]:
		num_tile_rows = np.ceil(lay_shape[0] / 9.0).astype(np.int32)
		num_tile_cols = np.ceil(lay_shape[1] / 9.0).astype(np.int32)
		num_tiles = num_tile_rows * num_tile_cols

		tile_start_idx = tile_count
		tile_end_idx = tile_count + num_tiles
		lay_promoters = promoters[:, tile_start_idx:tile_end_idx]
		tile_count = tile_end_idx

		templates = template_maker(lay_promoters, pop_size, num_tile_rows, num_tile_cols)
		matrices = nca_func(templates, pop_size, nca_kernels)
		lay_list.append(matrices[0, :, :lay_shape[0], :lay_shape[1]])

	return lay_list