if [[ -z "${BIDIRECTIONAL_API}" ]]; then
    echo "Normal server selected."
    python backends/hypercycle_server.py
else
    echo "Bidirectional server selected."
    python backends/bidirectional_server.py
fi
