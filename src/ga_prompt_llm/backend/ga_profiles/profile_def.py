from abc import ABC, abstractclassmethod
from typing import Any

from callbacks import Callbacks, EmptyCallbacks
from genetic_algorithm import GeneticAlgorithm


class ProfileDefinition(ABC):
    @abstractclassmethod
    def get_default_inputs(cls) -> dict[str, str]:
        ...

    @abstractclassmethod
    def get_genetic_algorithm(
        cls, params: dict[str, Any], callbacks: Callbacks = EmptyCallbacks()
    ) -> GeneticAlgorithm:
        ...
