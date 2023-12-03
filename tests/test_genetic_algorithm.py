import pytest
from chromosome import Chromosome
from llm import LLM, Bloom


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


@pytest.mark.parametrize("llm", [Bloom()])
def test_execute_llm(llm: LLM, chromosome_prompts: list[Chromosome]) -> None:
    llm.tokenizer_population(chromosome_prompts)
    solutions = llm(chromosome_prompts)

    assert solutions[0] == "I love you." and solutions[1] == "Good luck."
