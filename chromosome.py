from dataclasses import dataclass, field
from itertools import count


@dataclass
class Chromosome:
    parent_id: int | tuple[int, int] = field(default=0)  # 0 = root
    id: int = field(default_factory=count(1).__next__, init=False)
    prompt: str | None = field(default=None)
    score: str | None = field(default=None)

    def __str__(self) -> str:
        return str({"prompt": self.prompt, "score": self.score})


@dataclass
class FixedLengthChromosome(Chromosome):
    mutable_mask: list[bool] | None = field(
        default=None
    )  # each item indicate if the word can mutate or not
