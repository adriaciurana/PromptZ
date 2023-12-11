import os
from population import Population

RANDOM_SEED = 0
POPULATION_SIZE = 5
MIN_GENE = 4
MAX_GENE = 32

if __name__ == "__main__":
    # Path to sample text.
    txt_file = os.path.join("data", "cyanide_text.txt")
    
    # Instantiate population.
    population = Population(
        size=POPULATION_SIZE,
        min_gene=MIN_GENE,
        max_gene=MAX_GENE,
        file=txt_file,
        random_seed=RANDOM_SEED)
    
    # Get sample chromosome.
    sample_chromosomes = population[0]
    
    # Print keywords.
    print("Keywords")
    print(sample_chromosomes.get_keywords(), "\n")
    
    # Print initial prompt.
    print("Initial prompt")
    print(sample_chromosomes.get_initial_prompt(), "\n")
    
    # Print prompts
    print("Prompts")
    print(sample_chromosomes)