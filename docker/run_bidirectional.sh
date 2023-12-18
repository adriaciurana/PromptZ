docker run \
    --name team_error_404_ga_api_bi_exec \
    --rm \
    -t \
    -p 4003:4003 \
    -e BIDIRECTIONAL_API=true \
    --gpus all \
    team_error_404_ga