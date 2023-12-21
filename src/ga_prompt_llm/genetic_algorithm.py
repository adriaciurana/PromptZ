import heapq
import logging
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any
from typing import Counter as TCounter

import numpy as np
from callbacks import Callbacks, EmptyCallbacks
from chromosome import Chromosome
from evaluator import Evaluator
from generator import Generator
from llm import LLM
from population_creator import PopulationCreator
from qqdm import qqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

# Use as reference: https://towardsdatascience.com/genetic-algorithms-for-natural-language-processing-b055aa7c14e9

class GeneticAlgorithm:
    @dataclass
    class RuntimeConfig:
        max_population: int = field(default=10)
        topk_population: int = field(default=5)
        iterations: int = field(default=1000)
        generator_samples: int = field(default=10)

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

        @classmethod
        def from_dict(cls, obj: dict[str, Any]) -> "GeneticAlgorithm.RuntimeConfig":
            return cls(**obj)

    def __init__(
        self,
        llm: LLM,
        population_creator: PopulationCreator,
        generator: Generator,
        evaluator: Evaluator,
        callbacks: Callbacks = EmptyCallbacks(),
        *,
        stop_condition_max_iterations_same_best_chromosome=100,
        stop_condition_last_mean_scores_length=30,
        stop_condition_last_mean_scores_tol=1e-3,
    ) -> None:
        self._llm = llm
        self._population_creator = population_creator

        self._generator = generator
        self._evaluator = evaluator
        self._callbacks = callbacks

        self._stop_condition_max_iterations_same_best_chromosome = (
            stop_condition_max_iterations_same_best_chromosome
        )
        self._stop_condition_last_mean_scores_length = (
            stop_condition_last_mean_scores_length
        )
        self._stop_condition_last_mean_scores_tol = stop_condition_last_mean_scores_tol

    def _filter_population(
        self, population: list[Chromosome], max_population: int
    ) -> list[Chromosome]:
        def _filter(c: Chromosome) -> float:
            return c.score

        return heapq.nlargest(
            max_population,
            population,
            key=_filter,
        )

    def __call__(
        self,
        initial_prompt: str,
        target: str | None = None,
        objective: str | None = None,
        runtime_config: RuntimeConfig = RuntimeConfig(),
        *args: dict[str, Any],  # TODO: TBD
        **kwargs: dict[str, Any],  # TODO: TBD
    ) -> list[Chromosome]:
        if target is None:
            target = objective

            assert (
                objective is not None
            ), "If target is not defined, please define the objective and use the proper evaluator"

        pbar = qqdm(range(runtime_config.iterations), total=runtime_config.iterations)

        # 1. init the evaluator
        if init_evaluator:
            self._evaluator.init(self._llm, target)
        logging.info("Init evaluator. Done")

        # 2. init the generator
        # 1. init the evaluator
        if init_generator:
            self._generator.init(self._llm, target)
        logging.info("Init generator. Done")

        # 3. create the initial population
        population = self._population_creator(initial_prompt, target, self._generator)
        logging.info("Init population. Done")

        # 4. computed score for the initial population
        self._evaluator(population)
        logging.info("Executed evaluator on the initial population. Done")

        # Initialize hook
        self._callbacks.init(population)

        # Condition stop
        self._stop_condition_best_chromosome_counter: TCounter[int] = Counter()
        self._stop_condition_last_mean_scores = deque(
            maxlen=self._stop_condition_last_mean_scores_length
        )

        # 5. iterate over N iterations
        for iteration in pbar:
            # 6. Generate similar sentences
            variations = self._generator(population, k=runtime_config.generator_samples)
            population += variations
            logging.info(f"Generated {iteration} variation.")

            # 7. Evaluate current population
            self._evaluator(variations)
            logging.info(f"Evaluated {iteration} population.")

            # Send variations
            self._callbacks.generated(iteration, variations)

            # 8. Filter population
            old_population = population
            population = self._filter_population(
                population, runtime_config.max_population
            )

            # Send filtered
            self._callbacks.filtered_by_populations(
                iteration, old_population, population
            )

            logging.info(f"Filtering {iteration} population.")

            best_chromosome: Chromosome = max(population, key=lambda c: c.score)
            best_chromosome_dict = {
                k: getattr(best_chromosome, k) for k in best_chromosome.show
            }
            mean_score = np.mean([c.score for c in population])
            self._stop_condition_last_mean_scores.append(mean_score)

            # logging
            infos = {f"best-solution-{k}": v for k, v in best_chromosome_dict.items()}
            infos.update({"mean-score": mean_score})
            pbar.set_infos(infos)

            if self.evaluate_stop_condition(population, best_chromosome):
                break

        # 9. Filter population
        best_population = self._filter_population(
            population, runtime_config.topk_population
        )
        logging.info("Filtering last population.")

        self._callbacks.results(best_population)

        return best_population

    def evaluate_stop_condition(
        self, current_population: list[Chromosome], best_chromosome: Chromosome
    ):
        self._stop_condition_best_chromosome_counter[best_chromosome.id] += 1
        _, best_counter = self._stop_condition_best_chromosome_counter.most_common(1)[0]
        if best_counter > self._stop_condition_max_iterations_same_best_chromosome or (
            len(self._stop_condition_last_mean_scores)
            == self._stop_condition_last_mean_scores.maxlen
            and np.std(self._stop_condition_last_mean_scores)
            <= self._stop_condition_last_mean_scores_tol
        ):
            return True

        return False
