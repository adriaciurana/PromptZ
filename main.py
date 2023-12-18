from evaluator import BERTSimilarityEvaluator
from generator import KeywordGAGenerator, LLMSimilarSentencesGenerator, ComposerGenerator
from genetic_algorithm import GeneticAlgorithm
from llm import Phi2, M0
from population_creator import GeneratorPopulationCreator

ga = GeneticAlgorithm(
    llm=Phi2(),
    population_creator=GeneratorPopulationCreator(20),
    generator=ComposerGenerator([(LLMSimilarSentencesGenerator(), 0.5), (KeywordGAGenerator(), 0.5)]),
    evaluator=BERTSimilarityEvaluator(),
)

ga(
    initial_prompt="Greet me as your friend",
    target="Hello my enemy",
)
