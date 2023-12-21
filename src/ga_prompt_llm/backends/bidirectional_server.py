#!/usr/bin/env python
import json
import os
import sys
from copy import copy
from pathlib import Path
from typing import Any

import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.wsgi
from tornado.options import define, options

sys.path.append(str(Path(__file__).parent / "../"))

from callbacks import Callbacks
from chromosome import Chromosome
from evaluator import Evaluator
from ga_config import ConfigDefinition, load_config
from generator import Generator
from genetic_algorithm import GeneticAlgorithm
from llm import LLM
from population_creator import PopulationCreator
from utils import CacheWithRegister, Register

BI_PORT = os.environ.get("BI_PORT", 4003)
DEFAULT_RUNTIME_CONFIG = GeneticAlgorithm.RuntimeConfig().to_dict()

define("port", type=int, default=BI_PORT)

PROJECT_PATH = Path(__file__).parent / "../"

tornado.autoreload.start()
tornado.autoreload.watch(str(PROJECT_PATH / "frontend/index.html"))
tornado.autoreload.watch(str(PROJECT_PATH / "frontend/index.html"))


class Website(tornado.web.RequestHandler):
    def get(self):
        self.render(str(PROJECT_PATH / "frontend/index.html"))


class ScriptJS(tornado.web.RequestHandler):
    def get(self):
        f = open(PROJECT_PATH / "frontend/script.js", "r")
        self.write(f.read())
        f.close()


class WebsocketCallbacks(Callbacks):
    def __init__(self, connection: "WebsocketCommunication") -> None:
        super().__init__()
        self._connection = connection

    def init(self, population: list[Chromosome]) -> None:
        self._connection.write_message(
            json.dumps(
                {"operation": "init", "population": [c.to_dict() for c in population]}
            )
        )

    def generated(self, iteration: int, variations: list[Chromosome]) -> None:
        self._connection.write_message(
            json.dumps(
                {
                    "operation": "generated",
                    "iteration": iteration,
                    "variations": [c.to_dict() for c in variations],
                }
            )
        )

    def filtered(
        self, iteration: int, current_status_population: dict[int, bool]
    ) -> None:
        self._connection.write_message(
            json.dumps(
                {
                    "operation": "filtered",
                    "iteration": iteration,
                    "current_status_population": current_status_population,
                }
            )
        )

    def results(self, population: list[Chromosome]) -> None:
        self._connection.write_message(
            json.dumps(
                {
                    "operation": "results",
                    "population_ids": [c.id for c in population],
                }
            )
        )


def run(params: dict[str, Any], connection: "WebsocketCommunication"):
    runtime_config_dict: dict[str, Any] = copy(DEFAULT_RUNTIME_CONFIG)
    runtime_config_dict.update(params.get("runtime_config", {}))

    #
    runtime_config: GeneticAlgorithm.RuntimeConfig = (
        GeneticAlgorithm.RuntimeConfig.from_dict(runtime_config_dict)
    )

    config: ConfigDefinition = load_config(params["config_name"])
    genetic_algorithm: GeneticAlgorithm = config.get_genetic_algorithm(
        params,
        callbacks=WebsocketCallbacks(connection),
    )

    # Obtain the initial prompt (the first broad approximation provided by the user).
    initial_prompt: str = params["initial_prompt"]

    # Obtain the target (defined by the user).
    target: str = params["target"]

    # # Let's start the party! Run the algorithm!
    genetic_algorithm(
        initial_prompt=initial_prompt, target=target, runtime_config=runtime_config
    )


def get_default_inputs(params: dict[str, Any], connection: "WebsocketCommunication"):
    config: ConfigDefinition = load_config(params["config_name"])
    return config.get_default_inputs()


COMMANDS = {"run": run, "get_default_inputs": get_default_inputs}


class WebsocketCommunication(tornado.websocket.WebSocketHandler):
    clients = []

    def check_origin(self, origin):
        return True

    def open(self):
        # clients must be accessed through class object!!!
        WebsocketCommunication.clients.append(self)
        print("\nWebSocket opened")

    def on_message(self, message):
        message_json = json.loads(message)
        COMMANDS[message_json["cmd"]](message_json["params"], self)

    def on_close(self):
        print("WebSocket closed")

        # clients must be accessed through class object!!!
        WebsocketCommunication.clients.remove(self)


def main():
    tornado_app = tornado.web.Application(
        [
            ("/", Website),
            ("/websocket", WebsocketCommunication),
            ("/script.js", ScriptJS),
        ]
    )
    server = tornado.httpserver.HTTPServer(tornado_app)
    server.listen(options.port)
    print(f"Bidireccional server running in http://localhost:{BI_PORT}")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
