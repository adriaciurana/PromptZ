from fitness_score import BERTScore
from genetic_algorithm import GeneticAlgorithm, MatingPoolPolicy, PopulationPolicy
from llm import Bloom
from parents_policy import TournamentSelection
from variations import MixSentences, Noise, VariationsPolicy

ga = GeneticAlgorithm(
    llm=Bloom(),
    population_policy=PopulationPolicy(10, 2_000),
    parents_policy=TournamentSelection(),
    mating_pool_policy=MatingPoolPolicy(),
    variations_policy=VariationsPolicy(crossovers=[MixSentences()], mutators=[Noise()]),
    fitness_score=BERTScore(),
)

ga("TBD", iterations=10_000, topk_solutions=10)
