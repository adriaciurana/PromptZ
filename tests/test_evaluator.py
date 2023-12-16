from typing import Any

import pytest
from chromosome import Chromosome
from evaluator import BERTSimilarityEvaluator, Evaluator
from llm import LLM


@pytest.fixture()
def population():
    return [
        Chromosome(
            prompt="I'm happy.",
        ),
        Chromosome(
            prompt="I'm completely depressed.",
        ),
    ]


class MockLLM(LLM):
    def __init__(self, max_batch: int = 10, device: str = "cuda:0") -> None:
        super().__init__(max_batch, device)

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        return self.generate_from_prompt([c.prompt for c in population], params)

    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        return prompts


mock_llm = MockLLM()


@pytest.mark.parametrize("evaluator", [BERTSimilarityEvaluator()])
def test_execute_llm(evaluator: Evaluator, population: list[Chromosome]) -> None:
    evaluator.init(mock_llm, "I'm full of happiness.")
    evaluator(population)

    assert population[0].prompt > population[1].prompt
