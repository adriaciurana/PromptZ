import re
import copy
import random
from dataclasses import dataclass
import warnings

import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet as wn

import ga_utils
from llm_collections import LLM, Flan
from chromosome_test import Chromosome
    
@dataclass
class Population:
    """
    Population object for the chromosomes.
    
    Parameters
    ----------
    
    size : int, default = 100
        The size of the population, and the number of chromosomes inside this
        population.
        
    min_gene : int, default = 4
        The minimum number of gene in this population.
    
    max_gene : int, default = 128
        The maximum number of gene in this population.
        
    file : str, default = None
        If a TXT file is given, then the initialisation will be based on the
        given TXT file.
        
    ratio : tuple, default = (0.8, 0.2)
        If a file is given, this will be the initial mix of the special
        vocabulary and the default vocabulary.
    """
    def __init__(
            self,
            size: int = 100,
            min_gene: int = 4,
            max_gene: int = 32,
            prompt_generation_model: LLM = None,
            file: str = None,
            ratio: tuple = (0.8, 0.2),
            random_seed: int = None):
        # Random seed.
        if random_seed is not None:
            try:
                np.random.seed(random_seed)
                random.seed(random_seed)
            except Exception as e:
                print(e)
        
        # Attributes.
        self._size = size
        self._min_gene = min_gene
        self._max_gene = max_gene
        
        # Prompt generation model.
        if prompt_generation_model is None:
            self._prompt_generation_model = Flan()
        else:
            self._prompt_generation_model = prompt_generation_model
        
        # List of chromosomes.
        self._chromosomes = []
        
        # Instantiate interrogative list.
        self._interrogative_words = ["what", "which", "where", "how"]
        # Instantiate the default vocab.
        self._default_vocab = self._get_default_vocab()
        # Instantiate a special vocab.
        self._special_vocab = ()
        
        # If file is given.
        if file is not None:
            # True if suceeded, False if failed.
            self._special_vocab_flag = self._generate_special_vocab_from_file(
                file, ratio)
        else:
            # False for random.
            self._special_vocab_flag = False
        
        # Generate the chromosomes.
        self.generate_chromosomes()
    
    def __len__(self):
        return len(self._chromosomes)
    
    def __iter__(self):
        self._current = 0
        return self
    
    def __next__(self):
        if self._current < self._size:
            self._current += 1
            return self._chromosomes[self._current - 1]
        else:
            raise StopIteration
    
    def __getitem__(self, index):
        try:
            return self._chromosomes[index]
        except Exception as e:
            print(e)
    
    def get_chromosomes(self):
        return copy.deepcopy(self._chromosomes)
    
    def set_ratio(self, ratio: tuple = (0.8, 0.2)):
        if not isinstance(ratio, tuple):
            self._ratio = (0.8, 0.2)
            return None
        if len(ratio) != 2:
            warnings.warn(
                "Length of ratio is not 2. Ratio set to default.", Warning)
        elif ratio[0] + ratio[1] != 1:
            warnings.warn(
                "Sum of ratio is not 1. Ratio set to default.", Warning)
        else:
            self._ratio = ratio
            return None
        self._ratio = (0.8, 0.2)
    
    def _filter_stopwords(self, tokens):
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words("english"))
        
        return [word for word in tokens if not word.lower() in stop_words]
    
    def _filter_text_special_characters(self, text):
        return re.sub('[^A-Za-z0-9 ]+', '', text)
    
    def _filter_noun_adjectives(self, tokens):
        # Get the tagged tokens.
        try:
            tagged_tokens = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            tagged_tokens = nltk.pos_tag(tokens)
        # Filter the noun + adjectives.
        filtered = [token[0] for token in tagged_tokens if bool(
            re.search("NN", token[1])) or bool(re.search("JJ", token[1]))]
        # Return the filtered.
        return filtered
    
    def _get_default_vocab(self):
        try:
            test = list(wn.all_synsets('n'))
        except LookupError:
            nltk.download("wordnet")
            test = list(wn.all_synsets('n'))
        return tuple(word for synset in test for word in synset.lemma_names())
    
    def _generate_special_vocab_from_file(self, file, ratio):
        try:
            # Import text.
            text = ga_utils.read_txt(file=file, append_string=True)
            # If failed, return false.
            if text is None:
                warnings.warn(
                    "Cannot import file. Set to random population.", Warning)
                return False
            
            # Set ratio.
            self.set_ratio(ratio)
            
            # Filter special characters.
            text = self._filter_text_special_characters(text)
            
            # Tokenize the text.
            try:
                # Tokenize the text.
                tokens = nltk.word_tokenize(text)
            except LookupError:
                nltk.download("punkt")
                # Tokenize the text.
                tokens = nltk.word_tokenize(text)
            
            # Filter tokens from stopwords.
            tokens = self._filter_stopwords(tokens)
            
            # Special vocab.
            self._special_vocab = tuple(set(
                self._filter_noun_adjectives(tokens)))
            
            return True
        except Exception as e:
            print(e)
            return False
    
    def generate_chromosomes(self) -> None:
        # Create an array that contains each chromosome size.
        gene_sizes = np.random.randint(
            low=self._min_gene, high=self._max_gene+1, size=self._size)
        
        # Loop to create chromosomes.
        for i, gene_size in enumerate(gene_sizes):
            # Select one interrogative word at random.
            keywords = [random.choice(self._interrogative_words)]
            
            # If a special vocab is present.
            if self._special_vocab_flag:
               # Determine how many special vocab.
               n_special = int(gene_size - 1 * self._ratio[0])
               # Determine how many default vocab.
               n_default = int(gene_size - 1 - n_special)
               # Get keywords.
               keywords.extend(list(
                   random.sample(self._special_vocab, n_special)) + list(
                       random.sample(self._default_vocab, n_default)))
            # If only the default vocab.
            else:
                # Generate based on gene size.
                keywords.extend(
                    list(random.sample(self._default_vocab, gene_size)))
            
            # Create chromosome.
            self._chromosomes.append(
                Chromosome(
                    keywords=keywords,
                    prompt_generation_model=self._prompt_generation_model))