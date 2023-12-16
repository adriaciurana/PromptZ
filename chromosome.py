from dataclasses import dataclass, field


@dataclass
class Chromosome:
    chromosome_id: int | None = field(default=None)
    parent: tuple | None = field(default=None)
    prompt: str | None = field(default=None)
    score: str | None = field(default=None)

    def __str__(self) -> str:
        return str({"prompt": self.prompt, "score": self.score})


class KeywordsChromosome(Chromosome):
    keywords: tuple | None = field(default=None)

    def __str__(self) -> str:
        return str(
            {"keywords": self.keywords, "prompt": self.prompt, "score": self.score}
        )
