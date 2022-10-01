
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from numpy import ndarray, uint8, float16


I_DTYPE = uint8
F_DTYPE = float16

@dataclass
class Genome:
	settings: Dict[str, Any]
	genes: List[ndarray]

	@property
	def num_individuals(self) -> int:
		return self.genes[0].shape[0]

	@property
	def population_size(self) -> int:
		return self.settings["population size"]

	@property
	def num_leaders(self) -> int:
		return self.settings["number of leaders"]

	@property
	def select_dir(self) -> str:
		return self.settings["select by"]