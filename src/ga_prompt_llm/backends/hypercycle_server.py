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
from ga_config import ConfigDefinition, load_config
from generator import Generator
from genetic_algorithm import GeneticAlgorithm
from llm import LLM
from population_creator import PopulationCreator
from pyhypercycle_aim import JSONResponseCORS, SimpleServer, aim_uri
from utils import CacheWithRegister, Register

PORT = os.environ.get("PORT", 4012)
DEFAULT_RUNTIME_CONFIG = GeneticAlgorithm.RuntimeConfig().to_dict()


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
                        "config_name": "objective_cyanide_chatgpt",
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

        config: ConfigDefinition = load_config(request_json["config_name"])
        genetic_algorithm: GeneticAlgorithm = config.get_genetic_algorithm(
            request_json,
        )

        # Obtain the initial prompt (the first broad approximation provided by the user).
        initial_prompt: str = request_json["initial_prompt"]

        # Obtain the target (defined by the user).
        target: str = request_json["target"]

        # # Let's start the party! Run the algorithm!
        genetic_algorithm(
            initial_prompt=initial_prompt, target=target, runtime_config=runtime_config
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
