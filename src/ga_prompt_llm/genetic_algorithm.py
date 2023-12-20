import heapq
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

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
    ) -> None:
        self._llm = llm
        self._population_creator = population_creator

        self._generator = generator
        self._evaluator = evaluator
        self._callbacks = callbacks

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
        new_target: str | None = None,
        init_generator: bool = True,
        init_evaluator: bool = True,
        runtime_config: RuntimeConfig = RuntimeConfig(),
        *args: dict[str, Any],  # TODO: TBD
        **kwargs: dict[str, Any],  # TODO: TBD
    ) -> list[Chromosome]:
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
            pbar.set_infos(
                {f"best-solution-{k}": v for k, v in best_chromosome_dict.items()}
            )

        # 9. Filter population
        best_population = self._filter_population(
            population, runtime_config.topk_population
        )
        logging.info("Filtering last population.")

        self._callbacks.results(best_population)

        return best_population

class GeneticAlgorithmLauncher():

    def __init__(
        self,
        llm: LLM,
        population_creator: PopulationCreator,
        generator: Generator,
        evaluator: Evaluator,
        callbacks: Callbacks = EmptyCallbacks(),
        objective: str = "similarity",
        llm_objective: LLM = None
    ) -> None:
        self._llm = llm
        self._population_creator = population_creator

        self._generator = generator
        self._evaluator = evaluator
        self._callbacks = callbacks

        self._objective = objective
        self._llm_objective = llm_objective
    
    