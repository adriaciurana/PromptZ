from abc import ABC, abstractmethod

from chromosome import Chromosome


class Hooks(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init(self, population: list[Chromosome]) -> None:
        ...

    @abstractmethod
    def generated(self, iteration: int, variations: list[Chromosome]) -> None:
        ...

    def filtered_by_populations(
        self,
        iteration: int,
        old_population: list[Chromosome],
        new_population: list[Chromosome],
    ) -> None:
        current_status_population: dict[int, bool] = {
            c.id: False for c in old_population
        }

        for c in new_population:
            current_status_population[c.id] = True

        self.filtered(iteration, current_status_population)

    @abstractmethod
    def filtered(
        self, iteration: int, current_status_population: dict[int, bool]
    ) -> None:
        ...

    @abstractmethod
    def results(self, population: list[Chromosome]) -> None:
        ...


class EmptyHooks(Hooks):
    def init(self, population: list[Chromosome]) -> None:
        ...

    def generated(self, iteration: int, variations: list[Chromosome]) -> None:
        ...

    def filtered(
        self, iteration: int, current_status_population: dict[int, bool]
    ) -> None:
        ...

    def results(self, population: list[Chromosome]) -> None:
        ...
