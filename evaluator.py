import gc
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


class SimilarityEvaluator(Evaluator, ABC):
    @abstractmethod
    def get_features(self, text: str | list[str]) -> torch.Tensor:
        ...


@Register("Evaluator")
class BERTSimilarityEvaluator(SimilarityEvaluator):
    def __init__(self, device: str = "cuda:0", max_batch: int = 10) -> None:
        super().__init__()

        self.device = device if torch.cuda.is_available() else "cpu"
        self._similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        self._target_features = torch.Tensor | None
        self.max_batch = max_batch

    @batch_processing(AGGREGATE_TENSORS)
    def get_features(self, text: str | list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self._similarity_model.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)
        self._target_features = self.get_features(target)

    def _similarity(
        self, target: torch.Tensor, solutions: torch.Tensor
    ) -> torch.Tensor:
        # Target: 1 x dim
        # Solutions: N x dim
        return util.pytorch_cos_sim(target, solutions)[0]
        # return (target @ solutions.T)[0]

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

                    if (
                        self._target not in c.prompt
                        and c.output not in c.prompt
                        and c.prompt not in c.output
                    ) or self._llm.IS_NAIVE:
                        valid_outputs.append(output)
                        valid_nonscored_population.append(c)

                    else:
                        invalid_nonscored_population.append(c)

                if len(valid_outputs) > 0:
                    valid_outputs_features = self.get_features(valid_outputs)

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
class ObjectiveBasedEvaluator(Evaluator):
    def __init__(self, similarity_evaluator: SimilarityEvaluator, k: int) -> None:
        super().__init__()
        self._similarity_evaluator = similarity_evaluator
        self._k = k
        self._target_features: torch.Tensor | None = None

    def prepare_target(self, non_blackbox_llm: LLM, target: str) -> None:
        # 1. Compute the LLM options
        prompt = f"""Given the following objective: 
        {target}
        Generate an answer.
        """
        outputs = non_blackbox_llm.generate_from_prompt(self._k * [prompt])

        # 2. Compute the features
        self._target_features = self._similarity_evaluator.get_features(outputs)

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)

        assert (
            self._target_features is not None
        ), "Target features are not initialized, please run `prepare_target` before the GA."

    def _similarity(
        self, target: torch.Tensor, solutions: torch.Tensor
    ) -> torch.Tensor:
        # Take the best solution w.r.t all the targets
        # Target: M x dim
        # Solutions: N x dim
        return torch.max(util.pytorch_cos_sim(target, solutions), dim=0)[0]

    def __call__(self, population: list[Chromosome]) -> None:
        # Very similar than the BERTSimilarityEvaluator, but we have another _similarity function
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

                    if (
                        self._target not in c.prompt
                        and c.output not in c.prompt
                        and c.prompt not in c.output
                    ) or self._llm.IS_NAIVE:
                        valid_outputs.append(output)
                        valid_nonscored_population.append(c)

                    else:
                        invalid_nonscored_population.append(c)

                if len(valid_outputs) > 0:
                    valid_outputs_features = self._similarity_evaluator.get_features(
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
