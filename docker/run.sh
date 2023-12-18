docker run \
    --name team_error_404_ga_api_exec \
    --rm \
    -t \
    -p 4002:4002 \
    -e BIDIRECTIONAL_API=true \
    --gpus all \
    team_error_404_ga