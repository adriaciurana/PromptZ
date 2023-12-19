from random import sample
from typing import Iterator

from chromosome import Chromosome


class MatingPoolPolicy:
    def __init__(self):
        pass

    def __call__(
        self, parents: list[Chromosome], k: int = 10
    ) -> Iterator[tuple[Chromosome, Chromosome]]:
        for _ in range(k):
            parent_a, parent_b = sample(parents, k=2)
            yield parent_a, parent_b

        yield from ()
