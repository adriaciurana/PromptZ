from abc import ABC, abstractmethod
from random import choice
from typing import Iterator

import numpy as np

from chromosome import Chromosome


class CrossOver(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self, chromosome_a: Chromosome, chromosome_b: Chromosome
    ) -> Chromosome:
        ...


class Mutator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, chromosome: Chromosome) -> Chromosome:
        ...


class VariationsPolicy:
    def __init__(
        self,
        crossovers: list[CrossOver],
        mutators: list[Mutator],
        prob_to_crossover: float = 0.1,
        prob_to_mutate: float = 0.001,
    ) -> None:
        self._mutators = mutators
        self._crossovers = crossovers
        self._prob_to_crossover = prob_to_crossover
        self._prob_to_mutate = prob_to_mutate

    def __call__(
        self, pair_parents: Iterator[tuple[Chromosome, Chromosome]]
    ) -> Iterator[Chromosome]:
        for c_a, c_b in pair_parents:
            if np.random.rand() < self._prob_to_crossover:
                crossover_method = choice(self._crossovers)
                c_a, c_b = crossover_method(c_a, c_b)

            if np.random.rand() < self._prob_to_mutate:
                mutator_method = choice(self._mutators)
                c_a = mutator_method(c_a)

            if np.random.rand() < self._prob_to_mutate:
                c_b = mutator_method(c_b)

            yield c_a
            yield c_b

        yield from ()
