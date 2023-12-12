from abc import ABC, abstractmethod

from chromosome import Chromosome, FixedLengthChromosome
from classic.mating_pool import MatingPoolPolicy
from classic.parents import ParentsPolicy
from classic.variations import VariationsPolicy
from llm import LLM


class Generator(ABC):
    ChromosomeObject = Chromosome

    def __init__(self) -> None:
        self._llm: LLM | None = None
        self._target: LLM | None = None

    def init(self, llm: LLM, target: str) -> None:
        self._llm = llm
        self._target = target

    @abstractmethod
    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        ...

# class RouterGenerator(Generator):

#     def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
#         for c in population:
#             if isinstance(c, Calvin):
#                 self._generator[Cal]


class LLMSimilarSentencesGenerator(Generator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        assert self._llm is not None and self._target is not None
        candidate_prompts: list[str] = []
        replicated_ids: list[int] = []
        for c in population:
            replicated_ids += k * [c.id]
            candidate_prompts += k * [
                f"""
                Using the following text prompt:
                
                {c.prompt}

                Create a similar prompt that can be better if you have to answer the following text:
                
                {self._target}
            """
            ]

        return [
            self.ChromosomeObject(parent_id=c_id, prompt=prompt)
            for c_id, prompt in zip(
                replicated_ids,
                self._llm.generate_from_prompt(
                    prompts=candidate_prompts,
                    params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
                ),
            )
        ]


class ClassicGenerator(Generator):
    def ChromosomeObject(self, *args, **kwargs) -> FixedLengthChromosome:
        return FixedLengthChromosome(*args, **kwargs, mutable_mask=self._mutable_mask)

    def __init__(
        self,
        parents_policy: ParentsPolicy,
        mating_pool_policy: MatingPoolPolicy,
        variations_policy: VariationsPolicy,
        mutable_mask: list[bool] | None = None,
    ) -> None:
        super().__init__()
        self._mutable_mask = mutable_mask
        self._parents_policy = parents_policy
        self._mating_pool_policy = mating_pool_policy
        self._variations_policy = variations_policy

    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        # 1. Choose the population that can breed (tournament selection)
        # https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
        best_parents = self._parents_policy(population)

        # 2. Pair the parents
        # https://stats.stackexchange.com/questions/581426/how-pairs-of-actual-parents-are-formed-from-the-mating-pool-in-nsga-ii
        pair_parents = self._mating_pool_policy(best_parents, k=k)

        # 3. Variations
        return list(self._variations_policy(pair_parents))
