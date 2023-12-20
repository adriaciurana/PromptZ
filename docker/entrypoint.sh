#!/bin/bash

if [[ -z "${BIDIRECTIONAL_API}" ]]; then
    echo "Normal server selected."
    poetry run python backends/hypercycle_server.py
else
    echo "Bidirectional server selected."
    poetry run python backends/bidirectional_server.py
fi
