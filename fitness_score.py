from abc import ABC, abstractmethod

import torch


class FitnessScore:
    def __init__(self, target: str) -> None:
        self.target = target

    @abstractmethod
    def __call__(self, solutions: list[str]) -> torch.Tensor:
        ...


class BERTScore(FitnessScore):
    def __init__(self, target: str):
        super().__init__(target)
        self._similarity_model = None  # TODO: TBD
        self._target_features = self._similarity_model(target)

    def _similarity(
        self, target: torch.Tensor, solutions: torch.Tensor
    ) -> torch.Tensor:
        # Target: 1 x dim
        # Solutions: N x dim
        return (target @ solutions.T)[0]

    def __call__(self, solutions: list[str]) -> torch.Tensor:
        solutions_features = self._similarity_model(solutions)
        return self._similarity(self._target_features, solutions_features)
