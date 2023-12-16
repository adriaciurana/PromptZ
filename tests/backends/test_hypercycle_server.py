# content of conftest.py
import sys
from pathlib import Path

import pytest
import requests

PROJECT_PATH = Path(__file__).parent / "../../"
sys.path.append(str(PROJECT_PATH))

from backends.test_manual_hypercycle_server import (
    test_run_endpoint as test_manual_run_endpoint,
)
from xprocess import ProcessStarter


@pytest.fixture
def hypercycle_server(request, xprocess):
    class Starter(ProcessStarter):
        # startup pattern
        pattern = "Uvicorn running on http://0.0.0.0:4002"

        # command to start process
        args = ["python", request.config.rootdir / "backends/hypercycle_server.py"]

    # ensure process is running and return its logfile
    logfile = xprocess.ensure("hypercycle_server", Starter)

    conn = "http://localhost:4002"
    yield conn

    # clean up whole process tree afterwards
    xprocess.getinfo("hypercycle_server").terminate()


def test_run_endpoint(hypercycle_server: str):
    test_manual_run_endpoint(hypercycle_server)
    # # Test the prompt endpoint
    # initial_prompt = "Greet me as your friend"
    # target = "Hello my enemy"
    # response = requests.post(
    #     hypercycle_server + "/run",
    #     json={
    #         "runtime_config": {
    #             "max_population": 10,
    #             "topk_population": 5,
    #             "iterations": 3,
    #             "generator_samples": 10,
    #         },
    #         "llm": "M0",
    #         "population_creator": {
    #             "name": "LLMPopulationCreator",
    #             "params": {"num_samples": 10},
    #         },
    #         "generator": {
    #             "name": "LLMSimilarSentencesGenerator",
    #             "params": {},
    #         },
    #         "evaluator": {
    #             "name": "BERTSimilarityEvaluator",
    #             "params": {"max_batch": 10},
    #         },
    #         "initial_prompt": initial_prompt,
    #         "target": target,
    #     },
    # )
    # response_json = response.json()
    # print(response_json)

    # # Use pytest's assertion style
    # assert "prompt" in response_json, "Response does not contain 'prompt'"
    # assert "score" in response_json, "Response does not contain 'score'"
    # assert isinstance(response_json["prompt"], str), "'prompt' is not a string"
    # assert isinstance(response_json["score"], (int, float)), "'score' is not a number"
    # assert len(response_json["prompt"]) > 0, "'prompt' is empty"
