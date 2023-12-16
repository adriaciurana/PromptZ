import re
import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet as wn

from chromosome import Chromosome
from llm import LLM, Mistral


class Generator(ABC):
    def __init__(self) -> None:
        self._llm: LLM | None = None
        self._target: LLM | None = None

    def init(self, llm: LLM, target: str) -> None:
        self._llm = llm
        self._target = target

    @abstractmethod
    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        ...


class LLMSimilarSentencesGenerator(Generator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        assert self._llm is not None and self._target is not None
        candidate_prompts: list[str] = []
        for c in population:
            candidate_prompts += k * [
                f"""
                Using the following text:
                
                {c.prompt}

                Create a similar sentence that can be better if you have to answer something related with:
                
                {self._target}
            """
            ]

        return [
            Chromosome(prompt=prompt)
            for prompt in self._llm.generate_from_prompt(
                prompts=candidate_prompts,
                params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
            )
        ]

class KeywordGAGenerator(Generator):
    def __init__(self) -> None:
        super().__init__()
        
        # Set the min and max gene.
        self._min_gene = None
        self._max_gene = None
        
        # Get the default vocabulary.
        self._default_vocab = None
        
        # Setup the input vocabulary.
        self._input_vocab = None
    
    def init(self, llm: LLM, target:str, min_gene: int = 2, max_gene: int = 32) -> None:
        super().init(llm, target)
        
        # Set the min and max gene.
        self._min_gene = min_gene
        self._max_gene = max_gene
        
        # Get the default vocabulary.
        self._default_vocab = {
            "nouns": self._init_default_vocab(grammar_code='n'),
            "verbs": self._init_default_vocab(grammar_code='v'),
            "adj": self._init_default_vocab(grammar_code='a'),
        }
        
        # Setup the input vocabulary.
        self._input_vocab = None
        
    def __call__(self, population: list[Chromosome], k: int, reset=False) -> list[Chromosome]:
        assert self._llm is not None and self._target is not None
        # If population is empty, generate from scratch.
        if not population or reset:
            return self._generate_from_scratch(k)
        # Else, generate new population.
        else:
            return self._generate_new_generation(population, k)
    
    def _generate_from_scratch(self, k: int):
        # Create an array that contains each chromosome size.
        gene_sizes = np.random.randint(
            low=self._min_gene, high=self._max_gene, size=k)
        
        # List of keywords.
        keywords_list = []
        
        # Loop to create chromosomes.
        for i, gene_size in enumerate(gene_sizes):
            # If input vocab, use some from input vocab.
            if self.check_input_vocab():
                # Determine how many input vocab.
                n_input = int(np.random.random_sample() * float(gene_size))
                keywords = self._get_random_input_vocab(n_input)
                # Determine how many default vocab.
                n_default = int(gene_size - len(keywords))
                # Get keywords.
                keywords.extend(self._get_random_default_vocab(n_default))
            # Only from default vocab.
            else:
                # Generate based on gene size.
                keywords = self._get_random_default_vocab(n_default)
            # Append to list.
            keywords_list.append(keywords)
        
        initial_prompts = self._keywords_to_prompt(keywords_list)
        prompts = self._llm.generate_from_prompt(
            prompts=initial_prompts,
            params={"max_new_tokens": 40, "do_sample": True, "top_k": 50})
        
        return [Chromosome(keywords=keywords, prompt=re.sub('"""', "", prompt)) for keywords, prompt in zip(keywords_list, prompts)]
    
    def _generate_new_generation(self, population: list[Chromosome], k: int):
        scores = [chromosome.score.cpu().item() for chromosome in population]
        # Deal with negatives.
        if min(scores) < 0:
            scores = [score + abs(min(scores)) for score in scores]
        # Get normalised scores.
        scores_probability = [score / sum(scores) for score in scores]
        
        # Population.
        initial_population = len(population)
        
        for i_new in range(k):
            parents = np.random.choice(
                range(initial_population), size=2, p=scores_probability)
            p1 = population[parents[0]]
            p2 = population[parents[1]]
            
            new_chromosome = self._crossover(p1, p2)
            population.append(new_chromosome)
        
        return population
    
    def _crossover(self, p1: Chromosome, p2: Chromosome):
        gene_size = np.random.randint(low=self._min_gene, high=self._max_gene)
        p1_p2_comb = list(list(p1.keywords) + list(p2.keywords))
        if len(p1_p2_comb) < gene_size:
            gene_size = len(p1_p2_comb)
        keywords = random.sample(p1_p2_comb, gene_size)
        initial_prompt = self._keywords_to_prompt([keywords])
        prompt = self._llm.generate_from_prompt(
            prompts=initial_prompt,
            params={"max_new_tokens": 40, "do_sample": True, "top_k": 50})
        return Chromosome(keywords=keywords, prompt=prompt[0])
    
    def _keywords_to_prompt(self, keywords_list):
        initial_prompts = []
        for keywords in keywords_list:
            # Format.
            formatted_keywords = re.sub("_", " ", ", ".join(keywords)).lower()
            # Generate prompt.
            initial_prompt = f"Generate an LLM prompt input that contains the following keywords: {formatted_keywords}."
            # Add tokens for mistral.
            if isinstance(self._llm, Mistral):
                initial_prompt = f"<s>[INST]{initial_prompt}[/INST]"
            # Initial prompts.
            initial_prompts.append(initial_prompt)
        return initial_prompts
    
    def _get_random_default_vocab(self, n_sample: int, subset_list: list = ["nouns", "verbs", "adj"]):
        all_words = []
        for key in subset_list:
            all_words.extend(list(self._default_vocab[key]))
        return random.sample(list(set(all_words)), n_sample)
    
    def _get_random_input_vocab(self, n_sample: int, subset_list: list = None):
        if subset_list is None:
            all_words = []
            for value in self._input_vocab.values():
                all_words.extend(list(value))
            all_words = list(set(all_words))
            if len(all_words) < n_sample:
                n_sample=len(all_words)
        return random.sample(all_words, n_sample)
    
    def _get_sum_input_vocab(self):
        all_words = []
        for value in self._input_vocab.values():
            all_words.extend(list(value))
        
    def check_input_vocab(self):
        if self._input_vocab is None:
            return False
        else:
            return True
    
    def set_input_vocab(self, **kwargs):
        self._input_vocab = {}
        for key, value in kwargs.items():
            self._input_vocab[str(key)] = self._extract_vocab_from_text(value)
    
    def _extract_vocab_from_text(self, text):
        # Filter special characters.
        text = re.sub('[^A-Za-z0-9 ]+', '', text).lower()
        
        # Tokenize the text.
        try:
            # Tokenize the text.
            tokens = nltk.word_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            # Tokenize the text.
            tokens = nltk.word_tokenize(text)
        
        # Filter stopwords.
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if not word.lower() in stop_words]
        
        # Filter to just nouns, and adjectives.
        try:
            tagged_tokens = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            tagged_tokens = nltk.pos_tag(tokens)
        # Filter the noun + adjectives.
        tokens = [token[0] for token in tagged_tokens if bool(
            re.search("NN", token[1])) or bool(re.search("JJ", token[1]))]
        
        return tuple(set(tokens))
    
    def _init_default_vocab(self, grammar_code: str = 'n'):
        try:
            words = list(wn.all_synsets(grammar_code))
        except LookupError:
            nltk.download("wordnet")
            words = list(wn.all_synsets(grammar_code))
        return tuple(word for synset in words for word in synset.lemma_names())