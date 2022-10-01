from typing import List
import numpy as np

# from neurogenome.genome import Genome, I_DTYPE
from genome import Genome

I_DTYPE = np.int32
F_DTYPE = np.float32


def selection(population: Genome, results: List[float]) -> Genome:
	"""
	Метод отбора

	**************
	
	Parameters
	----------
	population : Genome
		геном популяции
	results : List[float]
		список с результатами (вознаграждениями) по каждой особи
	"""

	# сортировка и отбор лидеров по результатам
	num_leaders = population.num_leaders
	if population.select_dir == "max":
		leader_indices = np.argsort(np.array(results))[::-1][:num_leaders]
	else:
		leader_indices = np.argsort(np.array(results))[:num_leaders]

	promoters = population.genes[0][leader_indices].copy()
	kernels = population.genes[1][leader_indices].copy()

	return Genome(
		settings=population.settings,
		genes=[promoters, kernels]
	)

# метод скрещивания
def crossover(leaders: Genome) -> Genome:
	"""
	Метод кроссинговера
	**************
	
	Parameters
	----------
	leaders : Genome
		геном лидеров
	"""
	num_individuals = leaders.num_individuals
	population_size = leaders.population_size

	promoters = __gene_hybridization(leaders.genes[0].copy(), num_individuals, population_size)
	kernels = __gene_hybridization(leaders.genes[1].copy(), num_individuals, population_size)

	return Genome(
		settings=leaders.settings,
		genes=[promoters, kernels]
	)

# метод мутации
def gene_mutation(population: Genome, mu: float = 0.0, sigma: float = 10.0) -> None:
	"""
	Метод мутации генов
	**************
	
	Parameters
	----------
	population : Genome
		популяция
	mu : float
		медиана нормального распределения
	sigma : float
		стандартное отклонение нормального распределения
	"""
	num_individuals = population.num_individuals
	min_val = -256
	max_val = 257

	for genes in population.genes:
		selected_individuals = np.random.permutation(num_individuals)[:np.random.randint(1, num_individuals)]
		genes_selected_individuals = genes[selected_individuals].copy()

		# mutation_value = np.random.normal(mu, sigma, genes_selected_individuals.shape)
		# mutation_value = np.random.randint(min_val, max_val, size=genes_selected_individuals.shape)
		# mutation_value = np.random.randint(min_val, max_val, size=genes_selected_individuals.shape)
		# mutation_mask = np.random.randint(0, 2, size=genes_selected_individuals.shape)
		# genes[selected_individuals] = np.clip(np.around(genes_selected_individuals + mutation_value * mutation_mask), min_val, max_val).astype(I_DTYPE)
		# mutation_value = np.random.randint(min_val // 2, max_val // 2, size=genes_selected_individuals.shape)
		mutation_value = np.random.normal(mu, sigma, genes_selected_individuals.shape)
		# mutation_value = np.random.random(size=genes_selected_individuals.shape)
		# mutation_value = np.random.normal(mu, sigma, genes.shape)
		genes[selected_individuals] = np.clip(np.around(genes_selected_individuals + mutation_value), min_val, max_val).astype(I_DTYPE)
		# genes[selected_individuals] = np.clip(genes_selected_individuals + mutation_value, min_val, max_val)
		# genes[selected_individuals] = genes_selected_individuals + mutation_value
		# genes += mutation_value
		# mutation_value = np.random.randint(-10, 11, size=genes.shape)
		# genes = np.clip(np.around(genes + mutation_value), min_val, max_val).astype(I_DTYPE)
		# genes[selected_individuals] = (genes_selected_individuals + mutation_value).astype(I_DTYPE)

def matrix_mutation(population: Genome, new_shape_list: List[List]) -> None:
	"""
	Метод мутации размера целевой матрицы
	**************
	
	Parameters
	----------
	population : Genome
		популяция
	"""
	population.settings["matrix shape"] = new_shape_list

	shape_arr = np.array(new_shape_list)
	current_num_promoters = population.genes[0].shape[1]
	new_num_promoters = np.ceil((shape_arr[:,0] * shape_arr[:,1]).sum() / 9.0).astype(np.int32)
	if new_num_promoters > current_num_promoters:
		num_new_promoters = new_num_promoters - current_num_promoters
		new_promoters = np.random.randint(-256, 257, size=(population.genes[0].shape[0], num_new_promoters))
		# new_promoters = np.random.random(size=(population.genes[0].shape[0], num_new_promoters))
		# new_promoters = np.random.randn(population.genes[0].shape[0], num_new_promoters)
		population.genes[0] = np.concatenate([population.genes[0], new_promoters], axis=1)

def __gene_hybridization(genes: np.ndarray, num_individuals: int, population_size: int) -> np.ndarray:
	"""
	Метод гибридизации двух весовых тензоров
	**************
	
	Parameters
	----------
	genes : np.ndarray 
		массив генов
	"""
	# создание дочернего тензора
	child_genes = np.zeros((population_size, *genes.shape[1:]), dtype=F_DTYPE)

	a = np.linspace(0.2, 1.0, num=num_individuals)[::-1]
	b = a/a.sum()
	parents_indices = np.random.choice(num_individuals, population_size, p=b)

	individuals_indices = np.arange(num_individuals)

	for idx_ch, idx_p0 in enumerate(parents_indices):
		idx_p1 = individuals_indices[individuals_indices != idx_p0][np.random.randint(num_individuals-1)]
		mask = np.random.randint(2, size=(*genes.shape[1:],)).astype(I_DTYPE)
		child_genes[idx_ch] = genes[idx_p0] * mask + genes[idx_p1] * (mask ^ 1)

	# возвращаем дочерний тензор
	return child_genes