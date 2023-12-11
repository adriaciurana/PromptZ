from evaluator import BERTSimilarityEvaluator
from generator import LLMSimilarSentencesGenerator
from genetic_algorithm import GeneticAlgorithm
from llm import M0
from population_creator import LLMPopulationCreator

ga = GeneticAlgorithm(
    llm=M0(),
    population_creator=LLMPopulationCreator(20),
    generator=LLMSimilarSentencesGenerator(),
    evaluator=BERTSimilarityEvaluator(),
)
ga(
    initial_prompt="Hello my friend",
    target="Create a positive sentence greeting your friend",
)
