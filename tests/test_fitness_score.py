import pytest
from chromosome import Chromosome
from fitness_score import BERTScore, FitnessScore


@pytest.fixture()
def chromosome_prompts():
    return [
        Chromosome(
            prompt="",
            solution="I'm happy.",
        ),
        Chromosome(
            prompt="",
            solution="I'm completely depressed.",
        ),
    ]


@pytest.mark.parametrize("fitness_score", [BERTScore()])
def test_execute_llm(
    fitness_score: FitnessScore, chromosome_prompts: list[Chromosome]
) -> None:
    fitness_score.init("I'm full of happiness.")
    cmp = fitness_score([c.solution for c in chromosome_prompts])
    assert cmp[0] > cmp[1]
