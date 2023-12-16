from abc import ABC, abstractmethod
from typing import Any

from chromosome import Chromosome
from generator import Generator, KeywordGAGenerator


class PopulationCreator(ABC):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    @abstractmethod
    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        ...


class LLMPopulationCreator(PopulationCreator):
    def __init__(self, num_samples: int) -> None:
        super().__init__(num_samples)

    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        return generator([Chromosome(prompt=initial_prompt)], k=self.num_samples)

class KeywordsPopulationCreator(PopulationCreator):
    def __init__(self, num_samples: int) -> None:
        super().__init__(num_samples)
    
    def __call__(
        self, initial_prompt: str, target: str, generator: KeywordGAGenerator
    ) -> list[Chromosome]:
        # Set initial prompt.
        if not generator.check_input_vocab():
            generator.set_input_vocab(initial_prompt=initial_prompt, target=target)
        
        return generator([Chromosome(prompt=initial_prompt)], k=self.num_samples, reset=True)