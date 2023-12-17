import itertools
import random
import re
from abc import ABC, abstractmethod

import nltk
import numpy as np
from chromosome import Chromosome, FixedLengthChromosome, KeywordsChromosome
from classic.mating_pool import MatingPoolPolicy
from classic.parents import ParentsPolicy
from classic.variations import VariationsPolicy
from llm import LLM, Mistral
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from utils import Register


class Generator(ABC):
    ChromosomeObject = Chromosome

    def __init__(self) -> None:
        self._llm: LLM | None = None
        self._target: LLM | None = None

    def init(self, llm: LLM, target: str) -> None:
        self._llm = llm
        self._target = target

    @abstractmethod
    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        ...


@Register("Generator")
class LLMSimilarSentencesGenerator(Generator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        assert (
            is_initial and len(population) == 1
        ), "For the first population, you need to provide only one chromosome"
        assert self._llm is not None and self._target is not None
        candidate_prompts: list[str] = []
        replicated_ids: list[int] = []
        for c in population:
            replicated_ids += k * [c.id]
            candidate_prompts += k * [
                f"""
                Using the following text prompt:
                
                {c.prompt}

                Create a similar prompt that can be better if you have to answer the following text:
                
                {self._target}
            """
            ]

        return [
            self.ChromosomeObject(parent_id=c_id, prompt=prompt, by=id(self.__class__))
            for c_id, prompt in zip(
                replicated_ids,
                self._llm.generate_from_prompt(
                    prompts=candidate_prompts,
                    params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
                ),
            )
        ]


@Register("Generator")
class KeywordGAGenerator(Generator):
    ChromosomeObject = KeywordsChromosome

    def __init__(self, min_gene: int = 2, max_gene: int = 32) -> None:
        super().__init__()

        # Set the min and max gene.
        self._min_gene = min_gene
        self._max_gene = max_gene

        # Get the default vocabulary.
        self._default_vocab = {
            "nouns": self._init_default_vocab(grammar_code="n"),
            "verbs": self._init_default_vocab(grammar_code="v"),
            "adj": self._init_default_vocab(grammar_code="a"),
        }

        # Setup the input vocabulary.
        self._input_vocab = None

    def _generate_from_scratch(self, k: int):
        # Create an array that contains each chromosome size.
        gene_sizes = np.random.randint(low=self._min_gene, high=self._max_gene, size=k)

        # List of keywords.
        keywords_list = []

        # Loop to create chromosomes.
        for gene_size in gene_sizes:
            # If input vocab, use some from input vocab.
            # if self._input_vocab is None:
            # Determine how many input vocab.
            n_input = int(np.random.random_sample() * float(gene_size))
            keywords = self._get_random_input_vocab(n_input)

            # Determine how many default vocab.
            n_default = int(gene_size - len(keywords))

            # Get keywords.
            # TODO: Why extend?
            keywords += self._get_random_default_vocab(n_default)

            # Only from default vocab.
            # else:
            #     # TODO: n_default is not defined
            #     # Generate based on gene size.
            #     keywords = self._get_random_default_vocab(n_default)

            # Append to list.
            keywords_list.append(keywords)

        initial_prompts = self._keywords_to_prompt(keywords_list)
        prompts = self._llm.generate_from_prompt(
            prompts=initial_prompts,
            params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
        )

        return [
            self.ChromosomeObject(
                keywords=keywords,
                prompt=re.sub('"""', "", prompt),
                by=id(self.__class__),
            )
            for keywords, prompt in zip(keywords_list, prompts)
        ]

    def __call__(
        self, population: list[KeywordsChromosome], k: int, is_initial: bool = False
    ) -> list[KeywordsChromosome]:
        assert self._llm is not None and self._target is not None

        # Set initial prompt.
        if self._input_vocab is None:
            self.set_input_vocab(
                initial_prompt=population[0].prompt, target=self._target
            )

        if is_initial:
            assert len(population) == 1, "The initial population has to be equal to 1."
            return self._generate_from_scratch(k=k)

        # Otherwise, generate new population
        assert all(
            isinstance(c, KeywordsChromosome) for c in population
        ), "The next population can only be `KeywordsChromosome` type."
        return self._generate_new_generation(population, k)

    def _generate_new_generation(
        self, population: list[KeywordsChromosome], k: int
    ) -> list[KeywordsChromosome]:
        scores = [chromosome.score for chromosome in population]

        # Deal with negatives.
        if min(scores) < 0:
            min_abs_score = abs(min(scores))
            scores = [score + min_abs_score for score in scores]

        # Get normalised scores.
        scores_probability = [score / sum(scores) for score in scores]

        # Population.
        initial_population = len(population)

        for _ in range(k):
            parents: list[int] = np.random.choice(
                range(initial_population), size=2, p=scores_probability
            )
            p1 = population[parents[0]]
            p2 = population[parents[1]]

            new_chromosome = self._crossover(p1, p2)
            population.append(new_chromosome)

        return population

    def _crossover(
        self, p1: KeywordsChromosome, p2: KeywordsChromosome
    ) -> KeywordsChromosome:
        gene_size = np.random.randint(low=self._min_gene, high=self._max_gene)
        p1_p2_comb = list(list(p1.keywords) + list(p2.keywords))
        if len(p1_p2_comb) < gene_size:
            gene_size = len(p1_p2_comb)
        keywords = random.sample(p1_p2_comb, gene_size)
        initial_prompt = self._keywords_to_prompt([keywords])
        prompt = self._llm.generate_from_prompt(
            prompts=initial_prompt,
            params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
        )
        return self.ChromosomeObject(
            keywords=keywords, prompt=prompt[0], by=id(self.__class__)
        )

    def _keywords_to_prompt(self, keywords_list: list[list[str]]) -> list[str]:
        initial_prompts: list[str] = []
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

    def _get_random_default_vocab(
        self, n_sample: int, subset_list: list = ["nouns", "verbs", "adj"]
    ) -> list[str]:
        all_words = []

        for key in subset_list:
            all_words += list(self._default_vocab[key])

        return random.sample(list(set(all_words)), n_sample)

    def _get_random_input_vocab(
        self, n_sample: int, subset_list: list[str] | None = None
    ):
        # TODO (aciurana): What happens if subset_list is not None?
        if subset_list is None:
            all_words = []

            for value in self._input_vocab.values():
                all_words += list(value)

            all_words = list(set(all_words))

            if len(all_words) < n_sample:
                n_sample = len(all_words)

        return random.sample(all_words, n_sample)

    def _get_sum_input_vocab(self) -> list[str]:
        all_words = []
        for value in self._input_vocab.values():
            all_words += list(value)

        return all_words

    def set_input_vocab(self, **kwargs) -> None:
        self._input_vocab = {}
        for key, value in kwargs.items():
            self._input_vocab[str(key)] = self._extract_vocab_from_text(value)

    def _extract_vocab_from_text(self, text: str) -> tuple[str, ...]:
        # Filter special characters.
        text = re.sub("[^A-Za-z0-9 ]+", "", text).lower()

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
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))

        tokens = [word for word in tokens if not word.lower() in stop_words]

        # Filter to just nouns, and adjectives.
        try:
            tagged_tokens = nltk.pos_tag(tokens)

        except LookupError:
            nltk.download("averaged_perceptron_tagger")
            tagged_tokens = nltk.pos_tag(tokens)

        # Filter the noun + adjectives.
        tokens = [
            token[0]
            for token in tagged_tokens
            if bool(re.search("NN", token[1])) or bool(re.search("JJ", token[1]))
        ]

        return tuple(set(tokens))

    def _init_default_vocab(self, grammar_code: str = "n"):
        try:
            words = list(wn.all_synsets(grammar_code))

        except LookupError:
            nltk.download("wordnet")
            words = list(wn.all_synsets(grammar_code))

        return tuple(word for synset in words for word in synset.lemma_names())


@Register("Generator")
class ComposerGenerator(Generator):
    def __init__(self, generators: list[tuple[Generator, float]] | list[Generator]):
        generators_dict: dict[int, tuple[Generator, float]]
        if isinstance(generators[0], Generator):
            generators_dict = {
                id(g.__class__): (g, 1.0 / len(generators)) for g in generators.items()
            }

        else:
            w_sum = sum(w for _, w in generators)
            generators_dict = {id(g.__class__): (g, w / w_sum) for g, w in generators}

        self._generators = generators_dict

    @classmethod
    def _by_generator(cls, chromosome: Chromosome) -> str:
        return chromosome.by

    def init(self, llm: LLM, target: str) -> None:
        super().init(llm, target)
        for g, _ in self._generators.values():
            g.init(llm, target)

    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        if is_initial:
            init_population: list[Chromosome] = []
            for _, (generator, weight) in self._generators.items():
                init_population += generator(
                    population, k=int(weight * k), is_initial=True
                )

            return init_population

        else:
            new_variations: list[Chromosome] = []

            for key_group, population_group in itertools.groupby(
                population, self._by_generator
            ):
                generator, weight = self._generators[key_group]
                new_variations += generator(
                    list(population_group), k=int(weight * k), is_initial=True
                )

            return new_variations


@Register("Generator")
class ClassicGenerator(Generator):
    def ChromosomeObject(self, *args, **kwargs) -> FixedLengthChromosome:
        return FixedLengthChromosome(*args, **kwargs, mutable_mask=self._mutable_mask)

    def __init__(
        self,
        parents_policy: ParentsPolicy,
        mating_pool_policy: MatingPoolPolicy,
        variations_policy: VariationsPolicy,
        mutable_mask: list[bool] | None = None,
    ) -> None:
        super().__init__()
        self._mutable_mask = mutable_mask
        self._parents_policy = parents_policy
        self._mating_pool_policy = mating_pool_policy
        self._variations_policy = variations_policy

    def __call__(self, population: list[Chromosome], k: int) -> list[Chromosome]:
        # 1. Choose the population that can breed (tournament selection)
        # https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
        best_parents = self._parents_policy(population)

        # 2. Pair the parents
        # https://stats.stackexchange.com/questions/581426/how-pairs-of-actual-parents-are-formed-from-the-mating-pool-in-nsga-ii
        pair_parents = self._mating_pool_policy(best_parents, k=k)

        # 3. Variations
        return list(self._variations_policy(pair_parents, by=id(self.__class__)))
