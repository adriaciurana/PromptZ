from abc import ABC, abstractmethod
from typing import Any

from chromosome import Chromosome
from generator import Generator
from utils import Register


class PopulationCreator(ABC):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    @abstractmethod
    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        ...


@Register("PopulationCreator")
class LLMPopulationCreator(PopulationCreator):
    def __init__(self, num_samples: int) -> None:
        super().__init__(num_samples)

    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        return generator(
            [generator.ChromosomeObject(prompt=initial_prompt)], k=self.num_samples
        )