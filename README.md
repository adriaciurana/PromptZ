# OpenMesh Hackathon - Challenge 4: Genetic Algorithm Project

## Overview

This repository contains a project developed for Challenge 4: Genetic Algorithm in the OpenMesh Hackathon. The project provides various tools for optimizing prompts using a genetic algorithm (GA) approach.

### Features

- **GA Framework:** The project includes a framework based on genetic algorithms that facilitates prompt optimization.
- **Chromosome Definition:** The definition of chromosomes can be done using keywords `KeywordsChromosome` (similar to a conventional GA) or through a more advanced approach `Chromosome`. The advanced approach involves generating mutations using another Language Model (LLM), avoiding the fixed-length problem typically associated with GA chromosome genes. In this approach, the chromosome solely stores the prompt.
- **Generation Process:** The generation process depends on the chromosome definition. If the chromosome is keyword-based, mutations and mixing between parents are used to obtain new populations (`KeywordGAGenerator`). However, mutating a chromosome effectively becomes challenging when the solution space is vast, as is often the case in Natural Language Processing (NLP) problems. In constrast if is based on sentences, the generation will be produced by a LLM generating similar sentences (`LLMSimilarSentencesGenerator`).
- **Evaluator (Fitness Function):** The project incorporates an evaluator module, equivalent to a fitness function. This module assesses whether the outputs of the Language Model (LLM) meet a specific objective. Two versions of the evaluator have been implemented: one based on similarity between the target and the LLM output (`BERTSimilarityEvaluator`), utilizing a document retrieval model for comparison. Another intriguing approach is the use of an objective-based evaluator (`TODO`). In this approach, the task is defined not solely by the output but by the objective itself (e.g., generating harmful phrases, obtaining privileged information about a proprietary LLM). This allows analysis to prevent prompt injections in environments where extra security is required.
- **Backends:** At Error 404 Team, we prioritize usability as well. That's why we believe that the optimal way to analyze a Genetic Algorithm (GA) model isn't solely by relying on a backend that returns the best result. Since the project is experimental, the complete trace of the entire algorithm can provide a wealth of information regarding which words or sentences are prone to achieving better performance. This, in turn, enables drawing conclusions and gaining a deeper understanding of your Language Model's behavior. That's why we've incorporated two backends into our system. One is a RestAPI that implements the hypercycle server, and the other is a frontend/backend combination that enables bidirectional real-time interaction with the algorithm.

## Usage

To use this project, follow the steps below:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/adriaciurana/GA-prompt-LLM.git
   ```

2. Navigate to the project directory.
   ```bash
   cd GA-prompt-LLM
   ```

3. Explore the various tools and components provided in the repository.


## Install locally
You can use the the ``pyproject.toml` and `poetry` to install all the dependences.
1. `pip install poetry`.
2. `poetry install`
3. `poetry shell`
4. Now it's time to start the party! :dancer:

## Build a docker
You can use the following script to use a docker:
`./docker/build.sh`

## Run the backend
### Hypercycle backend
This setup allows you to execute our algorithm by making a POST call to a RestAPI (you can find the specifications [here](https://github.com/adriaciurana/GA-prompt-LLM/blob/main/src/ga_prompt_llm/backends/hypercycle_server.py#L34)). It returns the best result. You can also find an example [here](https://github.com/adriaciurana/GA-prompt-LLM/blob/main/src/ga_prompt_llm/backends/test_manual_hypercycle_server.py).

Youll notice that it has many parameters; this is because it enables the execution of the model, the generator, the evaluator, and the creation of the desired population.

Example:
```python
{
    "runtime_config": {
    "max_population": 10,
    "topk_population": 5,
    "iterations": 3,
    "generator_samples": 10,
    },
    "llm": "M0", # You can find more information about all the models here: https://github.com/adriaciurana/GA-prompt-LLM/blob/main/src/ga_prompt_llm/llm.py
    "population_creator": {
        "name": "GeneratorPopulationCreator", # You can find more information about all the populations creators here (for now, only `GeneratorPopulationCreator` is available): https://github.com/adriaciurana/GA-prompt-LLM/blob/main/src/ga_prompt_llm/population_creator.py
        "params": {"num_samples": 10},
    },
    "generator": {
        "name": "LLMSimilarSentencesGenerator", # You can find more information about all the generators here: https://github.com/adriaciurana/GA-prompt-LLM/blob/main/src/ga_prompt_llm/generator.py
        "params": {},
    },
    "evaluator": {
        "name": "BERTSimilarityEvaluator", # You can find more information about all the evaluators here: https://github.com/adriaciurana/GA-prompt-LLM/blob/main/src/ga_prompt_llm/evaluator.py
        "params": {"max_batch": 10},
    },
    "initial_prompt": initial_prompt, # the pseudo-initialization of the prompt (starting from a random or distant initialization from the objective requires a significantly higher number of iterations for the algorithm. Therefore, user-provided prior information is extremely useful as the `initial_prompt`, significantly influencing the algorithm's efficiency.)
    "target": target, # the target or the objective (in case that you select the `TODO`).
}
```

#### locally
`./scripts/run_backend.sh`


#### docker
`./docker/run_backend.sh`

### Frontend/Backend real-time interactive
As mentioned in the introduction, we firmly believe in usability and interpretability.

If we cannot interpret our results, how can we model our Language Model (LLM)? That's why we've included a small dashboard that allows real-time interpretation of the Genetic Algorithm (GA) results. This feature serves as the "cherry on the cake" for our solution.

This solution maintains a persistent connection with the server, enabling each advancement in the Genetic Algorithm (GA) to be visualized as a graph in real-time on a small dashboard. Additionally, real-time updates of the top-k best solutions are also included.

Visualization of the introduction website:
![First page](./docs/bi_1.png)

Realtime visualization of the topk best solutions + graph. 
In the graph, each level represents an iteration, each node corresponds to a chromosome, and the edges depict mutations or parent mixing that generates new chromosomes.

The colors have the following meanings:
- Red: Eliminated chromosomes.
- Blue: Chromosomes that form the current population.
- Green: Best-performing chromosomes.
- White: Solely indicative of the root.

![Realtime dashboard](./docs/bi_2.png)

You can also interact with the graph, and by clicking on a node, you can access more information about that chromosome. This can be observed in the bottom-right part of the following figure:

![Interactive Graph](./docs/bi_3.png)

#### locally
`./scripts/run_backend_bidirectional.sh`


#### docker
`./docker/run_backend_bidirectional.sh`

## API
TODO

## Contributors

- [Adrià Ciurana - Team Lead / AI Engineer / SW Engineer ](https://github.com/adriaciurana)
- [Calvin Joseph - AI Engineer / SW Engineer](https://github.com/calvindajoseph)
- [Ashraf - Product Designer / UX/UI Designer / Pitch Designer / Project Manager](https://github.com/uxashraf)
- [Shoaib Ahamed Shafi -Frontend Backend Engineer / SW Engineer](https://github.com/uxashraf)

## License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to the project and make it even more robust and versatile!