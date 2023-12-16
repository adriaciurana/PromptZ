"""
OpenDI HyperCycle Hackathon 2023
Challenge 3: Genetic Algorithm for Automated Prompt Engineering
"""
import os
import sys
from copy import copy
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent / "../"))

from chromosome import Chromosome
from evaluator import Evaluator
from generator import Generator
from genetic_algorithm import GeneticAlgorithm
from llm import LLM
from population_creator import PopulationCreator
from pyhypercycle_aim import JSONResponseCORS, SimpleServer, aim_uri
from utils import CacheWithRegister, Register

PORT = os.environ.get("PORT", 4002)
DEFAULT_RUNTIME_CONFIG = GeneticAlgorithm.RuntimeConfig().to_dict()
CACHED_LLMS: dict[str, LLM] = CacheWithRegister(
    "LLM",
    kwargs={"max_batch": 10, "device": "cuda:0", "result_length": 50},
)


class HypercycleServer(SimpleServer):
    manifest = {
        "name": "HypercycleServer",
        "short_name": "hypercycle-server",
        "version": "0.1",
        "license": "MIT",
        "author": "Error 404 Team",
    }

    def __init__(self) -> None:
        pass

    @aim_uri(
        uri="/run",
        methods=["POST"],
        endpoint_manifest={
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Returns the prompt and the score based on the desired output",
            "example_calls": [
                {
                    "body": {
                        "runtime_config": {
                            "max_population": 10,
                            "topk_population": 5,
                            "iterations": 3,
                            "generator_samples": 10,
                        },
                        "llm": "M0",
                        "population_creator": {
                            "name": "GeneratorPopulationCreator",
                            "params": {"num_samples": 10},
                        },
                        "generator": {
                            "name": "LLMSimilarSentencesGenerator",
                            "params": {},
                        },
                        "evaluator": {
                            "name": "BERTSimilarityEvaluator",
                            "params": {"max_batch": 10},
                        },
                        "initial_prompt": "Greet me as your friend",
                        "target": "Hello my enemy",
                    },
                    "method": "POST",
                    "query": "",
                    "headers": "",
                    "output": {
                        "prompt": "simple, lively, strong",
                        "score": 0.004,
                        "output": "long sentence",
                    },
                }
            ],
        },
    )
    async def prompt(self, request):
        request_json: dict[str, Any] = await request.json()

        runtime_config_dict: dict[str, Any] = copy(DEFAULT_RUNTIME_CONFIG)
        runtime_config_dict.update(request_json.get("runtime_config", {}))

        #
        runtime_config: GeneticAlgorithm.RuntimeConfig = (
            GeneticAlgorithm.RuntimeConfig.from_dict(runtime_config_dict)
        )

        # Use the cache to reload the same LLM.
        llm_name: str = request_json["llm"]
        llm: LLM = CACHED_LLMS[llm_name]

        # Create a population creator object.
        population_creator_json: str = request_json["population_creator"]
        population_creator: PopulationCreator = Register.get(
            "PopulationCreator", population_creator_json["name"]
        )(**population_creator_json["params"])

        # Create a population creator object.
        generator_json: str = request_json["generator"]
        generator: Generator = Register.get("Generator", generator_json["name"])(
            **generator_json["params"]
        )

        # Create a evaluator creator object.
        evaluator_json: str = request_json["evaluator"]
        evaluator: Evaluator = Register.get("Evaluator", evaluator_json["name"])(
            **evaluator_json["params"]
        )

        # Obtain the initial prompt (the first broad approximation provided by the user).
        initial_prompt: str = request_json["initial_prompt"]

        # Obtain the target (defined by the user).
        target: str = request_json["target"]

        # Create the GA object
        genetic_algorithm = GeneticAlgorithm(
            llm=llm,
            population_creator=population_creator,
            generator=generator,
            evaluator=evaluator,
        )

        # Let's start the party! Run the algorithm!
        chromosomes: list[Chromosome] = genetic_algorithm(
            initial_prompt=initial_prompt, target=target, runtime_config=runtime_config
        )
        return JSONResponseCORS(
            {
                "prompt": chromosomes[0].prompt,
                "score": chromosomes[0].score,
                "output": chromosomes[0].output,
            }
        )


def main():
    # example usage:
    app = HypercycleServer()
    app.run(uvicorn_kwargs={"port": PORT, "host": "0.0.0.0"})


if __name__ == "__main__":
    main()
