from evaluator import BERTSimilarityEvaluator, ObjectiveBasedEvaluator
from generator import KeywordGAGenerator, LLMSimilarSentencesGenerator
from genetic_algorithm import GeneticAlgorithm
from llm import M0
from population_creator import GeneratorPopulationCreator

target = "Hello my enemy"
big_llm = M0()
evaluator = ObjectiveBasedEvaluator(
    similarity_evaluator=BERTSimilarityEvaluator(), k=100
)
evaluator.prepare_target(target=target, non_blackbox_llm=big_llm)
del big_llm

m0 = M0()
ga = GeneticAlgorithm(
    llm=m0,
    population_creator=GeneratorPopulationCreator(20),
    generator=LLMSimilarSentencesGenerator(),
    evaluator=evaluator,
)

ga(
    initial_prompt="Greet me as your friend",
    target=target,
)
