from evaluator import BERTSimilarityEvaluator, ObjectiveBasedEvaluator
from generator import KeywordGAGenerator, LLMSimilarSentencesGenerator
from genetic_algorithm import GeneticAlgorithm
from llm import M0
from population_creator import GeneratorPopulationCreator

m0 = M0()
ga = GeneticAlgorithm(
    llm=m0,
    population_creator=GeneratorPopulationCreator(20),
    generator=LLMSimilarSentencesGenerator(),
    evaluator=ObjectiveBasedEvaluator(
        similarity_evaluator=BERTSimilarityEvaluator(), non_blackbox_llm=m0, k=100
    ),
)

ga(
    initial_prompt="Greet me as your friend",
    target="Hello my enemy",
)
