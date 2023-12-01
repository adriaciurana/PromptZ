from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from chromosome import Chromosome


class ParentsPolicy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, population: list[Chromosome]) -> Iterator[Chromosome]:
        ...


class TournamentSelection(ParentsPolicy):
    def __init__(self, k: int = 3, num_parents: int = 100) -> None:
        super().__init__()
        self.k = k
        self.num_parents = num_parents

    def __call__(self, population: list[Chromosome]) -> Iterator[Chromosome]:
        for _ in range(self.num_parents):
            tournament_chromosomes = np.choice(self.k, len(population), replace=False)
            best_idx = max(
                tournament_chromosomes, key=lambda idx: population[idx].score
            )
            yield population[best_idx]

        yield from ()
