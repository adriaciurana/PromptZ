from abc import ABC, abstractmethod

from chromosome import Chromosome
from llm import LLM


class Generator(ABC):
    def __init__(self) -> None:
        self._llm: LLM | None = None
        self._target: LLM | None = None

    def init(self, llm: LLM, target: str) -> None:
        self._llm = llm
        self._target = target

    @abstractmethod
    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        ...


class LLMSimilarSentencesGenerator(Generator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        assert self._llm is not None and self._target is not None
        candidate_prompts: list[str] = []
        for c in population:
            candidate_prompts += k * [
                f"""
                Using the following text:
                
                {c.prompt}

                Create a similar sentence that can be better if you have to answer something related with:
                
                {self._target}
            """
            ]

        return [
            Chromosome(prompt=prompt)
            for prompt in self._llm.generate_from_prompt(
                prompts=candidate_prompts,
                params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
            )
        ]
