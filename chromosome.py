from dataclasses import dataclass

import torch


@dataclass
class Chromosome:
    __slots__ = ["prompt", "tokens", "solution", "score"]

    def __init__(
        self,
        prompt: str,
        tokens: torch.Tensor | None = None,
        solution: str | None = None,
        score: float | None = None,
    ) -> None:
        self.prompt = prompt
        self.tokens = tokens
        self.solution = solution
        self.score = score

    def __str__(self) -> str:
        return str(
            {"prompt": self.prompt, "score": self.score, "solution": self.solution}
        )
