import heapq
from abc import ABC, abstractmethod
from random import sample

from mutators import Mutator
from qqdm import qqdm

from chromosome import Chromosome
from llm import LLM
from parents_policy import ParentsPolicy
from variations import VariationsPolicy

# Use as reference: https://towardsdatascience.com/genetic-algorithms-for-natural-language-processing-b055aa7c14e9


def compute_fitness_func(
    llm: LLM, population: list[Chromosome], fitness_score: FitnessScore
):
    population_without_score = [c for c in population if c.score is None]
    for start_idx in range(0, len(population_without_score), llm.max_batch):
        end_idx = min(len(population), start_idx + llm.max_batch)
        batch_population = population[start_idx:end_idx]
        batch_solutions = llm(batch_population)
        batch_scores = fitness_score(batch_solutions)
        for c, solution, score in zip(batch_population, batch_solutions, batch_scores):
            c.solution = solution
            c.score = score

    return population


class PopulationPolicy:
    def __init__(self, init_population: int, max_population, int) -> None:
        self.init_population = init_population
        self.max_population = max_population

    def init(self, target: str) -> list[Chromosome]:
        # TODO: TBD
        return []


class MatingPoolPolicy:
    def __init__(self, k: int = 10):
        self.k = k

    def __init__(
        self, parents: list[Chromosome]
    ) -> Iterator[tuple[Chromosome, Chromosome]]:
        for iter_idx in range(self.k):
            parent_a, parent_b = sample(parents, k=2)
            yield parent_a, parent_b

        yield from ()


class FitnessScore:
    def __init__(self, target: str):
        self._similarity_model = None  # TBD
        self._target_features = self._similarity_model(target)

    def _similarity(
        self, target: torch.Tensor, solutions: torch.Tensor
    ) -> torch.Tensor:
        # Target: 1 x dim
        # Solutions: N x dim
        return (target @ solutions.T)[0]

    def __call__(self, solutions: list[str]) -> torch.Tensor:
        solutions_features = self._similarity_model(solutions)
        return self._similarity(self._target_features, solutions_features)


class GeneticAlgorithm:
    def __init__(
        self,
        llm: LLM,
        population_policy: PopulationPolicy,
        parents_policy: ParentsPolicy,
        mating_pool_policy: MatingPoolPolicy,
        variations_policy: VariationsPolicy,
        fitness_score: FitnessScore,
    ) -> None:
        self._llm = llm

        self._population_policy = population_policy
        self._parents_policy = parents_policy
        self._mating_pool_policy = mating_pool_policy
        self._variations_policy = variations_policy
        self._fitness_score = fitness_score

    def __call__(
        self,
        target: str,
        iterations: int,
        topk_solutions: int = 10,
        *args: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> None:
        pbar = qqdm(range(iterations))
        population = self._population_policy.init(target)
        batch_tokens = self._llm.tokenize_population(population)
        # 1. per each chromosome, we need to compute the LLM output and the fitness score function
        compute_fitness_func(self._llm, population, self._fitness_score)

        for iteration in pbar:
            # 2. Choose the population that can breed (tournament selection)
            # https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
            best_parents = self._parents_policy(population)

            # 3. Pair the parents
            # https://stats.stackexchange.com/questions/581426/how-pairs-of-actual-parents-are-formed-from-the-mating-pool-in-nsga-ii
            pair_parents = self._mating_pool_policy(best_parents)

            # 4. Variations
            population += list(self._variations_policy(pair_parents))

            # 5. Filter population
            population = headq.nlargest(
                self._population_policy.max_population,
                key=lambda c: c.score,
            )

            # 6. Compute scores
            compute_fitness_func(self._llm, population, self._fitness_score)

            pbar.set_infos(
                {"best-solution-score": max(population, key=lambda c: c.score)},
            )

        # 7. Filter population
        best_population = headq.nlargest(
            topk_solutions,
            key=lambda c: c.score,
        )

        return best_population
