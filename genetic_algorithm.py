import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

from chromosome import Chromosome
from evaluator import Evaluator
from generator import Generator
from hooks import EmptyHooks, Hooks
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

    def __init__(
        self,
        llm: LLM,
        population_creator: PopulationCreator,
        generator: Generator,
        evaluator: Evaluator,
        hooks: Hooks = EmptyHooks(),
    ) -> None:
        self._llm = llm
        self._population_creator = population_creator

        self._generator = generator
        self._evaluator = evaluator
        self._hooks = hooks

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
        target: str,
        runtime_config: RuntimeConfig = RuntimeConfig(),
        *args: dict[str, Any],  # TODO: TBD
        **kwargs: dict[str, Any],  # TODO: TBD
    ) -> None:
        pbar = qqdm(range(runtime_config.iterations), total=runtime_config.iterations)

        # 1. init the evaluator
        self._evaluator.init(self._llm, target)
        logging.info("Init evaluator. Done")

        # 2. init the generator
        self._generator.init(self._llm, target)
        logging.info("Init generator. Done")

        # 3. create the initial population
        population = self._population_creator(initial_prompt, target, self._generator)
        logging.info("Init population. Done")

        # 4. computed score for the initial population
        self._evaluator(population)
        logging.info("Init fitness score. Done")

        # Initialize hook
        self._hooks.init(population)

        # 5. iterate over N iterations
        for iteration in pbar:
            # 6. Generate similar sentences
            variations = self._generator(population, k=runtime_config.generator_samples)
            population += variations
            logging.info(f"Generated {iteration} variation.")

            # 7. Evalute current population
            self._evaluator(population)
            logging.info(f"Evaluaated {iteration} population.")

            # Send variations
            self._hooks.generated(iteration, variations)

            # 8. Filter population
            old_population = population
            population = self._filter_population(
                population, runtime_config.max_population
            )

            # Send filtered
            self._hooks.filtered_by_populations(iteration, old_population, population)

            logging.info(f"Filtering {iteration} population.")

            best_chromosome: Chromosome = max(population, key=lambda c: c.score)
            pbar.set_infos(
                {
                    "best-solution-score": float(best_chromosome.score),
                    "best-solution-prompt": best_chromosome.prompt.strip(),
                },
            )

        # 9. Filter population
        best_population = self._filter_population(
            population, runtime_config.topk_population
        )
        logging.info("Filtering last population.")

        self._hooks.results(best_population)

        return best_population
