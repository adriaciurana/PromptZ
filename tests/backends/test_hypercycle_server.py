# content of conftest.py
import sys
from pathlib import Path

import pytest
import requests
from backend.test_manual_hypercycle_server import run_endpoint_for_testing
from xprocess import ProcessStarter


@pytest.fixture
def hypercycle_server(request, xprocess):
    class Starter(ProcessStarter):
        # startup pattern
        pattern = "Uvicorn running on http://0.0.0.0:4002"

        # command to start process
        args = [
            "python",
            request.config.rootdir / "src/ga_prompt_llm/backend/hypercycle_server.py",
        ]

    # ensure process is running and return its logfile
    logfile = xprocess.ensure("hypercycle_server", Starter)

    conn = "http://localhost:4002"
    yield conn

    # clean up whole process tree afterwards
    xprocess.getinfo("hypercycle_server").terminate()


def test_run_endpoint(hypercycle_server: str):
    run_endpoint_for_testing(hypercycle_server)
