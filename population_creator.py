from abc import ABC, abstractmethod
from typing import Any

from chromosome import Chromosome, KeywordsChromosome
from generator import Generator, KeywordGAGenerator


class PopulationCreator(ABC):
    COMPATIBLE_GENERATORS: list[str] = []

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    @abstractmethod
    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        ...


class SimilarityPopulationCreator(PopulationCreator):
    COMPATIBLE_GENERATORS: list[str] = ["LLMSimilarSentencesGenerator"]

    def __init__(self, num_samples: int) -> None:
        super().__init__(num_samples)

    def __call__(
        self, initial_prompt: str, target: str, generator: Generator
    ) -> list[Chromosome]:
        return generator(
            [Chromosome(prompt=initial_prompt)], target=target, k=self.num_samples
        )


# class KeywordsPopulationCreator(PopulationCreator):
#     COMPATIBLE_GENERATORS: list[str] = ["KeywordGAGenerator"]

#     def __init__(self, num_samples: int) -> None:
#         super().__init__(num_samples)

#     def __call__(
#         self, initial_prompt: str, target: str, generator: KeywordGAGenerator
#     ) -> list[KeywordsChromosome]:
#         return generator.from_scratch(
#             k=self.num_samples, initial_prompt=initial_prompt, target=target
#         )
