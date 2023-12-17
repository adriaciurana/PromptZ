from evaluator import BERTSimilarityEvaluator
from generator import KeywordGAGenerator, LLMSimilarSentencesGenerator
from genetic_algorithm import GeneticAlgorithm
from llm import Mistral
from population_creator import GeneratorPopulationCreator

ga = GeneticAlgorithm(
    llm=Mistral(),
    population_creator=GeneratorPopulationCreator(20),
    generator=KeywordGAGenerator(),
    evaluator=BERTSimilarityEvaluator(),
)
# ga(
#     initial_prompt="Greet me as your enemy",
#     target="Hello my friend",
# )

ga(
    initial_prompt="Greet me as your friend",
    target="Hello my enemy",
)
