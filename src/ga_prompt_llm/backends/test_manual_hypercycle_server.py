import requests


def run_endpoint_for_testing(url: str = "http://localhost:4002"):
    # Test the prompt endpoint
    initial_prompt = "Greet me as your friend"
    target = "Hello my enemy"
    response = requests.post(
        url + "/run",
        json={
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
            "initial_prompt": initial_prompt,
            "target": target,
        },
    )
    response_json = response.json()
    print(response_json)

    # Use pytest's assertion style
    assert "prompt" in response_json, "Response does not contain 'prompt'"
    assert "output" in response_json, "Response does not contain 'output'"
    assert "score" in response_json, "Response does not contain 'score'"
    assert isinstance(response_json["prompt"], str), "'prompt' is not a string"
    assert isinstance(response_json["output"], str), "'output' is not a string"
    assert isinstance(response_json["score"], (int, float)), "'score' is not a number"
    assert len(response_json["prompt"]) > 0, "'prompt' is empty"
    assert len(response_json["output"]) > 0, "'output' is empty"


if __name__ == "__main__":
    run_endpoint_for_testing()
