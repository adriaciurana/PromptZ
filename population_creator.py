from abc import ABC, abstractmethod

from chromosome import Chromosome
from generator import Generator
from utils import Register


class PopulationCreator(ABC):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    @abstractmethod
    def __call__(self, initial_prompt: str, generator: Generator) -> list[Chromosome]:
        ...


@Register("PopulationCreator")
class GeneratorPopulationCreator(PopulationCreator):
    def __init__(self, num_samples: int) -> None:
        super().__init__(num_samples)

    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        return generator(
            [Chromosome(prompt=initial_prompt)], k=self.num_samples, is_initial=True
        )
