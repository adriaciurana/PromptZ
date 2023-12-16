import logging
from abc import ABC, abstractmethod

import torch
from chromosome import Chromosome
from llm import LLM
from sentence_transformers import SentenceTransformer, util
from transformers.utils import logging
from utils import AGGREGATE_TENSORS, DisableLogger, Register, batch_processing

logger = logging.get_logger("transformers")
logger.setLevel(logging.ERROR)


class Evaluator:
    def __init__(self) -> None:
        self._llm: LLM | None = None
        self._target: LLM | None = None

    def init(self, llm: LLM, target: str) -> None:
        self._llm = llm
        self._target = target

    @abstractmethod
    def __call__(self, population: list[Chromosome]) -> None:
        ...


@Register("Evaluator")
class BERTSimilarityEvaluator(Evaluator):
    def __init__(self, device: str = "cuda:0", max_batch: int = 10) -> None:
        super().__init__()

        self.device = device if torch.cuda.is_available() else "cpu"
        self._similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        self._target_features = torch.Tensor | None
        self.max_batch = max_batch

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)
        with torch.no_grad():
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

    @batch_processing(AGGREGATE_TENSORS)
    def _similarity_model_encode(self, prompts: list[str]) -> torch.Tensor:
        return self._similarity_model.encode(
            prompts, convert_to_tensor=True, show_progress_bar=False
        )

    def __call__(self, population: list[Chromosome]) -> None:
        assert self._llm is not None
        assert self._target is not None
        assert self._target_features is not None

        nonscored_population: list[Chromosome] = []
        for c in population:
            if c.score is None:
                nonscored_population.append(c)

        if len(nonscored_population) == 0:
            return

        with DisableLogger():
            with torch.no_grad():
                outputs = self._llm(nonscored_population)
                for o, c in zip(outputs, nonscored_population):
                    c.output = o

                outputs_features = self._similarity_model_encode(outputs)

            scores = self._similarity(self._target_features, outputs_features)
        for c, score in zip(
            nonscored_population,
            scores,
        ):
            c.score = float(score)
