#!/bin/bash

if [[ -z "${BIDIRECTIONAL_API}" ]]; then
    echo "Normal server selected."
    poetry run python backend/hypercycle_server.py
else
    echo "Bidirectional server selected."
    poetry run python backend/bidirectional_server.py
fi
