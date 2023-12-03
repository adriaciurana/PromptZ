from abc import ABC, abstractmethod

import torch
from sentence_transformers import SentenceTransformer, util


class FitnessScore(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init(self, target: str) -> None:
        ...

    @abstractmethod
    def __call__(self, solutions: list[str]) -> torch.Tensor:
        ...


class BERTScore(FitnessScore):
    def __init__(self, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self._similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )

    def init(self, target: str) -> None:
        self._target_features = self._similarity_model.encode(
            target, convert_to_tensor=True
        )

    def _similarity(
        self, target: torch.Tensor, solutions: torch.Tensor
    ) -> torch.Tensor:
        # Target: 1 x dim
        # Solutions: N x dim
        return util.pytorch_cos_sim(target, solutions)[0]
        # return (target @ solutions.T)[0]

    def __call__(self, solutions: list[str]) -> torch.Tensor:
        solutions_features = self._similarity_model.encode(
            solutions, convert_to_tensor=True
        )
        return self._similarity(self._target_features, solutions_features)
