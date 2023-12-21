import re
import textwrap
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
    def __init__(self, max_batch: int = 10) -> None:
        super().__init__()

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


class SimilarityFunction:
    def __init__(self):
        ...

    def init(self, similarity_evaluator: "SimilarityEvaluator") -> None:
        ...

    def __call__(self, target: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
        # Target: 1 x dim
        # Solutions: N x dim
        return util.pytorch_cos_sim(target, solutions)
        # output: 1 x N
        # return (target @ solutions.T)[0]


class NonDesiredSimilarityFunction(SimilarityFunction):
    def __init__(
        self,
        nondesired_sentences: list[str] = [],
        threshold: float = 0.75,
        similarity_function: SimilarityFunction = SimilarityFunction(),
    ):
        super().__init__()
        assert (
            len(nondesired_sentences) > 0
        ), "At least you need one non-desired sentence."
        self._similarity_function = similarity_function
        self._nondesired_sentences = nondesired_sentences
        self._threshold = threshold
        self._nondesired_sentences_features: torch.Tensor | None = None

    def init(self, similarity_evaluator: "SimilarityEvaluator") -> None:
        self._nondesired_sentences_features = similarity_evaluator.get_features(
            self._nondesired_sentences
        )
        self._similarity_function.init(similarity_evaluator)

    def __call__(self, target: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
        # 1. check if its close to a nondesired sentence
        nondesired_scores = torch.max(
            self._similarity_function(self._nondesired_sentences_features, solutions),
            dim=0,
        )[0]

        invalid_solutions = nondesired_scores >= self._threshold
        scores = self._similarity_function(target, solutions)
        scores[0, invalid_solutions] = -1.0
        return scores


class ObjectiveSimilarityFunction(SimilarityFunction):
    def __init__(
        self,
        non_blackbox_llm: LLM,
        objective: str,
        k: int,
        similarity_function: SimilarityFunction = SimilarityFunction(),
    ):
        super().__init__()
        self._similarity_function = similarity_function
        self._objective = objective
        self._k = k
        self._target_features: torch.Tensor | None = None

        # Compute the targets from the objective
        objective_prompt = textwrap.dedent(
            f"""
            Given the following objective: 
            {objective}
            Generate an answer.
            """
        )
        self._generated_targets_from_the_objective = (
            non_blackbox_llm.generate_from_prompt(self._k * [objective_prompt])
        )

    def init(self, similarity_evaluator: "SimilarityEvaluator") -> None:
        self._target_features = similarity_evaluator.get_features(
            self._generated_targets_from_the_objective
        )
        self._similarity_function.init(similarity_evaluator)

    def __call__(self, target: torch.Tensor, solutions: torch.Tensor) -> torch.Tensor:
        # Consider only the targets that the LLM creates from the objective
        scores = torch.max(
            self._similarity_function(self._target_features, solutions),
            dim=0,
        )
        return scores


@Register("Evaluator")
class SemanticSimilarityEvaluator(SimilarityEvaluator):
    def __init__(
        self,
        device: str = "cuda:0",
        max_batch: int = 10,
        similarity_function: SimilarityFunction = SimilarityFunction(),
    ) -> None:
        super().__init__()

        self.device = device if torch.cuda.is_available() else "cpu"
        self._similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=self.device
        )
        self._target_features = torch.Tensor | None
        self.max_batch = max_batch

        self._remove_nonvalid_words = re.compile(r"[^A-Za-z0-9 ]+")

        self._similarity_function = similarity_function
        self._similarity_function.init(self)

    @batch_processing(AGGREGATE_TENSORS)
    def get_features(self, text: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self._similarity_model.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)
        self._target_features = self.get_features([target])
        self._clean_target = self._remove_nonletters(target)

    def _remove_nonletters(self, txt: str):
        return self._remove_nonvalid_words.sub("", txt).lower()

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
                    valid_outputs_features = self.get_features(valid_outputs)
                    valid_scores = self._similarity_function(
                        self._target_features, valid_outputs_features
                    )[0]

                else:
                    valid_scores: list[float] = []

        for c in invalid_nonscored_population:
            c.score = -1.0

        for c, score in zip(
            valid_nonscored_population,
            valid_scores,
        ):
            c.score = float(score)
