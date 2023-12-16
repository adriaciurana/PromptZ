from dataclasses import asdict, dataclass, field
from itertools import count
from typing import Any, ClassVar


@dataclass
class Chromosome:
    show: ClassVar[list[str]] = ["prompt", "output", "score"]

    parent_id: int | tuple[int, int] = field(default=0)  # 0 = root
    id: int = field(default_factory=count(1).__next__, init=False)
    prompt: str | None = field(default=None)
    output: str | None = field(default=None)
    score: str | None = field(default=None)

    def __str__(self) -> str:
        return str({"prompt": self.prompt, "score": self.score})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Chromosome":
        return cls(**obj)


class KeywordsChromosome(Chromosome):
    show: ClassVar[list[str]] = ["keywords", "prompt", "output", "score"]

    keywords: tuple | None = field(default=None)

    def __str__(self) -> str:
        return str(
            {"keywords": self.keywords, "prompt": self.prompt, "score": self.score}
        )


@dataclass
class FixedLengthChromosome(Chromosome):
    mutable_mask: list[bool] | None = field(
        default=None
    )  # each item indicate if the word can mutate or not
