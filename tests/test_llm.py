import pytest
from chromosome import Chromosome
from llm import LLM, M0


@pytest.fixture()
def chromosome_prompts():
    return [
        Chromosome(
            prompt="Translate to English: Je t’aime.",
        ),
        Chromosome(
            prompt="Translate to English: Bonne chance.",
        ),
    ]


@pytest.mark.parametrize("llm", [M0()])
def test_execute_llm(llm: LLM, chromosome_prompts: list[Chromosome]) -> None:
    solutions = llm(chromosome_prompts)

    assert solutions[0] == "I love you." and solutions[1] == "Good luck."
