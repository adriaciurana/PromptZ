import re
from abc import ABC, abstractmethod

import numpy as np
import torch
from chromosome import Chromosome
from llm import LLM
from sentence_transformers import SentenceTransformer, util
from transformers.utils import logging as logging_t
from utils import AGGREGATE_TENSORS, DisableLogger, Register, batch_processing

logger = logging_t.get_logger("transformers")
logger.setLevel(logging_t.ERROR)


class Evaluator(ABC):
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
class MockEvaluator(Evaluator):
    def __init__(self, device: str = "cuda:0", max_batch: int = 10) -> None:
        super().__init__()

        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_batch = max_batch

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)

    def __call__(self, population: list[Chromosome]) -> None:
        assert self._llm is not None
        assert self._target is not None

        nonscored_population: list[Chromosome] = []
        for c in population:
            if c.score is None:
                c.output = "Mock output for: " + c.prompt
                nonscored_population.append(c)

        if len(nonscored_population) == 0:
            return

        for c in nonscored_population:
            c.score = 2 * np.random.rand() - 1  # Simulate cosine similarity


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

        self._remove_re = re.compile(r"[^A-Za-z0-9 ]+")

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)
        with torch.no_grad():
            self._target_features = self._similarity_model.encode(
                target, convert_to_tensor=True
            )
        self._clean_target = self._remove_nonletters(target)

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

    def _remove_nonletters(self, txt: str):
        return self._remove_re.sub("", txt).lower()

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

        # invalid: contains:
        #   - target inside prompt.
        #   - output inside prompt.
        #   - prompt inside output.
        valid_nonscored_population: list[Chromosome] = []
        invalid_nonscored_population: list[Chromosome] = []
        valid_outputs: list[str] = []
        with DisableLogger():
            with torch.no_grad():
                outputs = self._llm(nonscored_population)
                for output, c in zip(outputs, nonscored_population):
                    c.output = output

                    clean_output = self._remove_nonletters(c.output)
                    clean_prompt = self._remove_nonletters(c.prompt)

                    if (
                        self._clean_target not in clean_prompt
                        and clean_output not in clean_prompt
                        and clean_prompt not in clean_output
                    ) or self._llm.IS_NAIVE:
                        valid_outputs.append(output)
                        valid_nonscored_population.append(c)

                    else:
                        invalid_nonscored_population.append(c)

                if len(valid_outputs) > 0:
                    valid_outputs_features = self._similarity_model_encode(
                        valid_outputs
                    )

                    valid_scores = self._similarity(
                        self._target_features, valid_outputs_features
                    )

                else:
                    valid_scores: list[torch.Tensor] = []

        for c in invalid_nonscored_population:
            c.score = -1.0

        for c, score in zip(
            valid_nonscored_population,
            valid_scores,
        ):
            c.score = float(score)

@Register("Evaluator")
class BERTSimilarityObjectiveEvaluator(BERTSimilarityEvaluator):
    def __init__(self, device: str = "cuda:0", max_batch: int = 10) -> None:
        super().__init__(device=device, max_batch=max_batch)
    
    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)
        with torch.no_grad():
            self._target_features = self._similarity_model.encode(
                target, convert_to_tensor=True
            )
        