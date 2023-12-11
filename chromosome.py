from dataclasses import dataclass, field


@dataclass
class Chromosome:
    prompt: str | None = field(default=None)
    score: str | None = field(default=None)

    def __str__(self) -> str:
        return str({"prompt": self.prompt, "score": self.score})
