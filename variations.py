from abc import ABC, abstractmethod
from itertools import zip_longest
from random import choice
from typing import Iterator

import numpy as np
import torch
from chromosome import Chromosome


class CrossOver(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self, chromosome_a: Chromosome, chromosome_b: Chromosome
    ) -> Chromosome:
        ...


class MixSentences(CrossOver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, chromosome_a: Chromosome, chromosome_b: Chromosome
    ) -> Chromosome:
        tokens_a = chromosome_a.tokens
        tokens_b = chromosome_b.tokens

        tokens_mixed = []
        for t_a, t_b in zip(tokens_a, tokens_b):
            if np.random.rand() < 0.5:
                tokens_mixed.append(t_a)
            else:
                tokens_mixed.append(t_b)

        tokens_mixed = torch.stack(tokens_mixed, dim=0)

        return Chromosome(tokens=tokens_mixed)


class Mutator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, chromosome: Chromosome) -> Chromosome:
        ...


class Noise(ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, chromosome: Chromosome) -> Chromosome:
        tokens = chromosome.tokens.clone()
        rand_prob = torch.rand(*tokens.shape) > 0.5
        tokens[rand_prob] = torch.randint(0, 32_000, [rand_prob.sum()])

        return Chromosome(tokens=tokens)


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
                c_a = crossover_method(c_a, c_b)
                c_b = crossover_method(c_b, c_a)

            if np.random.rand() < self._prob_to_mutate:
                mutator_method = choice(self._mutators)
                c_a = mutator_method(c_a)

            if np.random.rand() < self._prob_to_mutate:
                mutator_method = choice(self._mutators)
                c_b = mutator_method(c_b)

            yield c_a
            yield c_b

        yield from ()
