from evaluator import BERTSimilarityEvaluator
from generator import LLMSimilarSentencesGenerator, KeywordGAGenerator
from genetic_algorithm import GeneticAlgorithm
from llm import M0, Mistral
from population_creator import LLMPopulationCreator, KeywordsPopulationCreator

ga = GeneticAlgorithm(
    llm=M0(),
    population_creator=KeywordsPopulationCreator(20),
    generator=KeywordGAGenerator(),
    evaluator=BERTSimilarityEvaluator(),
)
ga(
    initial_prompt="Hello my friend",
    target="Create a positive sentence greeting your friend",
)
