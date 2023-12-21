import itertools
import random
import re
import textwrap
from abc import ABC, abstractmethod
from collections import Counter
from random import sample
from typing import Callable

import nltk
import numpy as np
import spacy
from chromosome import Chromosome, FixedLengthChromosome, KeywordsChromosome
from classic.mating_pool import MatingPoolPolicy
from classic.parents import ParentsPolicy, TournamentSelection
from classic.variations import LLMCrossOver, LLMMutator, VariationsPolicy
from llm import LLM, Mistral, Phi2, Solar, OpenAIAPILLM
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from spacy.cli.download import download as spacy_download
from utils import Register


class Generator(ABC):
    ChromosomeObject = Chromosome

    def __init__(self) -> None:
        self.llm: LLM | None = None
        self.target: LLM | None = None

    def init(self, llm: LLM, target: str) -> None:
        self.llm = llm
        self.target = target

    @abstractmethod
    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        ...


@Register("Generator")
class MockGenerator(Generator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        if is_initial:
            assert (
                len(population) == 1
            ), "For the first population, you need to provide only one chromosome"
        assert self.llm is not None and self.target is not None

        return [
            self.ChromosomeObject(
                parent_id=c.id,
                prompt=f"sample_{c_idx}/" + c.prompt,
                by=id(self.__class__),
            )
            for c_idx, c in enumerate(population)
            for _ in range(k)
        ]


@Register("Generator")
class KeywordGAGenerator(Generator):
    ChromosomeObject = KeywordsChromosome

    def __init__(self, min_gene: int = 2, max_gene: int = 16) -> None:
        super().__init__()

        # Set the min and max gene.
        self._min_gene = min_gene
        self._max_gene = max_gene

        # Get the default vocabulary.
        self._default_vocab = {
            "nouns": self._init_default_vocab(grammar_code="n"),
            "adj": self._init_default_vocab(grammar_code="a"),
        }

        # Setup the input vocabulary.
        self._input_vocab = None

        # Setup most common vocab.
        self._common_vocab = None

        # Spacy model.
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            spacy_download("en_core_web_sm")
        self._spacy_model = spacy.load("en_core_web_sm")

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
            n_input = int(random.uniform(0.2, 1.0) * float(gene_size))
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
        prompts = self._generate_prompts(initial_prompts)

        population = [
            self.ChromosomeObject(
                keywords=keywords,
                prompt=re.sub('"""', "", prompt),
                by=id(self.__class__),
                parent_id=0,
            )
            for keywords, prompt in zip(keywords_list, prompts)
        ]

        # Update common vocab.
        self._update_common_vocab(population)

        return population

    def __call__(
        self, population: list[KeywordsChromosome], k: int, is_initial: bool = False
    ) -> list[KeywordsChromosome]:
        assert self.llm is not None and self.target is not None

        # Set initial prompt.
        if self._input_vocab is None:
            self.set_input_vocab(
                initial_prompt=population[0].prompt, target=self.target
            )

        if is_initial:
            assert len(population) == 1, "The initial population has to be equal to 1."
            return self._generate_from_scratch(k=k)

        # Otherwise, generate new population
        assert all(
            isinstance(c, KeywordsChromosome) for c in population
        ), "The next population can only be `KeywordsChromosome` type."
        return self._generate_new_generation(population, k)

    def _generate_prompts(self, initial_prompts):
        prompts = self.llm.generate_from_prompt(
            prompts=initial_prompts,
            params={"max_new_tokens": 100, "do_sample": True, "top_k": 50},
        )

        if isinstance(self.llm, Phi2):
            prompts = [
                re.sub(initial_prompt, "", prompt)
                for initial_prompt, prompt in zip(initial_prompts, prompts)
            ]
        elif isinstance(self.llm, Solar):
            prompts = [
                re.sub(re.escape(initial_prompt), "", prompt)
                for initial_prompt, prompt in zip(initial_prompts, prompts)
            ]

        return prompts

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

        # List for new generation.
        new_generation = []

        for _ in range(k):
            replace = False if initial_population > 1 else True
            parents: list[int] = np.random.choice(
                range(initial_population), size=2, p=scores_probability, replace=replace
            )
            p1 = population[parents[0]]
            p2 = population[parents[1]]

            new_chromosome = self._crossover(p1, p2)

            new_generation.append(new_chromosome)

        # Update common vocab.
        self._update_common_vocab(population + new_generation)

        # Return new generation.
        return new_generation

    def _crossover(
        self, p1: KeywordsChromosome, p2: KeywordsChromosome
    ) -> KeywordsChromosome:
        gene_size = np.random.randint(low=self._min_gene, high=self._max_gene)
        p1_p2_comb = list(list(p1.keywords) + list(p2.keywords))
        if len(p1_p2_comb) < gene_size:
            gene_size = len(p1_p2_comb)
        keywords = random.sample(p1_p2_comb, gene_size)
        initial_prompt = self._keywords_to_prompt([keywords])
        prompt = self._generate_prompts(initial_prompt)

        if p1 == p2:
            parent_id = p1.id
        else:
            parent_id = (p1.id, p2.id)

        new_chromosome = self.ChromosomeObject(
            keywords=tuple(set(keywords)),
            prompt=prompt[0],
            by=id(self.__class__),
            parent_id=parent_id,
        )

        # Mutation 1.
        probability_1 = 0.05
        if random.random() < probability_1:
            new_chromosome = self._mutation_1(new_chromosome)

        # Mutation 2.
        probability_2 = 0.05
        if random.random() < probability_2:
            new_chromosome = self._mutation_2(new_chromosome)

        # Mutation 3.
        probability_3 = 0.1
        if random.random() < probability_3:
            new_chromosome = self._mutation_3(new_chromosome)

        # Mutation 4.
        probability_4 = 0.03
        if random.random() < probability_4:
            new_chromosome = self._mutation_4(new_chromosome)

        return new_chromosome

    def _keywords_to_prompt(self, keywords_list: list[list[str]]) -> list[str]:
        initial_prompts: list[str] = []
        for keywords in keywords_list:
            # Format.
            formatted_keywords = re.sub("_", " ", ", ".join(keywords)).lower()

            # Generate prompt.
            initial_prompt = f"Generate an LLM prompt input that contains the following keywords: {formatted_keywords}."

            # Add tokens for mistral.
            if isinstance(self.llm, Mistral):
                initial_prompt = f"<s>[INST]{initial_prompt}[/INST]"
            if isinstance(self.llm, OpenAIAPILLM):
                initial_prompt += " Do not use the word 'lawyer'."
                initial_prompt += f" Make the prompt so the output of that prompt goal's is: {self.target}"

            # Initial prompts.
            initial_prompts.append(initial_prompt)

        return initial_prompts

    def _get_random_default_vocab(
        self, n_sample: int, subset_list: list = ["nouns", "adj"]
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
        else:
            return random.sample(subset_list, n_sample)

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

    def _update_common_vocab(self, population: list[Chromosome]):
        # Get all current keywords.
        keywords = [
            keyword for chromosome in population for keyword in chromosome.keywords
        ]
        # Determine top n.
        top_n = 20 if len(keywords) > 20 else len(keywords)
        # Update common vocab.
        self._common_vocab = [
            word for word, count in Counter(keywords).most_common(top_n)
        ]

    def _mutation_1(self, chromosome: Chromosome):
        # Swap some genes to the default vocab.
        n_random = np.random.randint(
            low=1, high=5 if len(chromosome.keywords) > 4 else len(chromosome.keywords)
        )

        # Indexes.
        indexes = np.random.choice(
            np.arange(len(chromosome.keywords)), size=n_random, replace=False
        )

        # Get the same amount of text from default vocab.
        new_keywords = self._get_random_default_vocab(n_sample=n_random)

        # Get keywords as list.
        current_keywords = list(chromosome.keywords)

        # Swap words.
        for index, new_keyword in zip(indexes, new_keywords):
            current_keywords[index] = new_keyword

        # Remove duplicates and put it back.
        chromosome.keywords = tuple(set(current_keywords))

        # Return chromosome.
        return chromosome

    def _mutation_2(self, chromosome: Chromosome):
        # Swap some genes to the input vocab.
        n_random = np.random.randint(
            low=1, high=4 if len(chromosome.keywords) > 3 else len(chromosome.keywords)
        )

        # Indexes.
        indexes = np.random.choice(
            np.arange(len(chromosome.keywords)), size=n_random, replace=False
        )

        # Get the same amount of text from input vocab.
        new_keywords = self._get_random_input_vocab(n_sample=n_random)

        # Get keywords as list.
        current_keywords = list(chromosome.keywords)

        # Swap words.
        for index, new_keyword in zip(indexes, new_keywords):
            current_keywords[index] = new_keyword

        # Remove duplicates and put it back.
        chromosome.keywords = tuple(set(current_keywords))

        # Return chromosome.
        return chromosome

    def _mutation_3(self, chromosome: Chromosome):
        # Swap some genes to the input vocab.
        n_random = np.random.randint(
            low=1, high=4 if len(chromosome.keywords) > 3 else len(chromosome.keywords)
        )

        # Indexes.
        indexes = np.random.choice(
            np.arange(len(chromosome.keywords)), size=n_random, replace=False
        )

        # Get the same amount of text from input vocab.
        new_keywords = random.sample(self._common_vocab, n_random)

        # Get keywords as list.
        current_keywords = list(chromosome.keywords)

        # Swap words.
        for index, new_keyword in zip(indexes, new_keywords):
            current_keywords[index] = new_keyword

        # Remove duplicates and put it back.
        chromosome.keywords = tuple(set(current_keywords))

        # Return chromosome.
        return chromosome

    def _mutation_4(self, chromosome: Chromosome):
        # Swap some genes to the input vocab.
        n_random = np.random.randint(
            low=1, high=4 if len(chromosome.keywords) > 3 else len(chromosome.keywords)
        )

        # Indexes.
        indexes = np.random.choice(
            np.arange(len(chromosome.keywords)), size=n_random, replace=False
        )

        # Get keywords as list.
        current_keywords = list(chromosome.keywords)

        # Get the synonyms.
        for index in indexes:
            word = current_keywords[index]
            synonyms = {
                lemma.name()
                for synset in wn.synsets(word)
                for lemma in synset.lemmas()
                if lemma.name() != word
            }
            if synonyms:
                current_keywords[index] = re.sub(
                    "-", "_", random.choice(list(synonyms))
                )

        # Remove duplicates.
        chromosome.keywords = tuple(set(current_keywords))

        # Return chromosome.
        return chromosome


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
                    list(population_group), k=int(weight * k), is_initial=False
                )

            return new_variations


@Register("Generator")
class ClassicGenerator(Generator):
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

        self._variations_policy.register(self)

    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        if is_initial:
            return list(
                self._variations_policy.init(population, by=id(self.__class__), k=k)
            )

        # 1. Choose the population that can breed (tournament selection)
        # https://en.wikipedia.org/wiki/Tournament_selection#:~:text=Tournament%20selection%20is%20a%20method,at%20random%20from%20the%20population.
        best_parents = list(self._parents_policy(population))

        # 2. Pair the parents
        # https://stats.stackexchange.com/questions/581426/how-pairs-of-actual-parents-are-formed-from-the-mating-pool-in-nsga-ii
        pair_parents = self._mating_pool_policy(best_parents, k=k)

        # 3. Variations
        return list(self._variations_policy(pair_parents, by=id(self.__class__)))


@Register("Generator")
class LLMSimilarSentencesGeneratorNotEfficient(Generator):
    # TODO: Redefine the API and merge with the classic generator
    # Not efficient if you want to use batch
    def __init__(self, num_parents: int = 100) -> None:
        self._internal_generator: Generator = ClassicGenerator(
            parents_policy=TournamentSelection(num_parents),
            mating_pool_policy=MatingPoolPolicy(),
            variations_policy=VariationsPolicy(
                crossovers=[LLMCrossOver()], mutators=[LLMMutator()]
            ),
        )

    @property
    def llm(self) -> LLM:
        return self._internal_generator.llm

    @property
    def target(self) -> LLM:
        return self._internal_generator.target

    def init(self, llm: LLM, target: str) -> None:
        self._internal_generator.init(llm, target)

    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        return self._internal_generator(population, k, is_initial=is_initial)


@Register("Generator")
class LLMSimilarSentencesGenerator(Generator):
    def __init__(self):
        super().__init__()

    def __call__(
        self, population: list[Chromosome], k: int, is_initial: bool = False
    ) -> list[Chromosome]:
        if is_initial:
            return self._mutate(population, k=k)

        # Mixin and mutate by the prob.
        to_mixup: list[tuple[Chromosome, Chromosome]] = []
        to_mutate: list[Chromosome] = []

        p = np.array([c.score for c in population])
        p += p.min()
        p /= sum(p)
        indices = list(range(len(population)))

        # Mixup
        parents_pool = np.random.choice(indices, size=k, replace=True, p=p).tolist()
        for _ in range(k // 2):
            c_a_id, c_b_id = sample(parents_pool, k=2)
            to_mixup.append((population[c_a_id], population[c_b_id]))

        new_variations = self._mixup(to_mixup, k=1)

        # Mutate
        for _ in range(k - k // 2):
            selected_to_mutate_id = np.random.choice(indices, p=p)
            to_mutate.append(population[selected_to_mutate_id])

        new_variations += self._mutate(to_mutate, k=1)

        return new_variations

    def _generate_by_callback(
        self,
        chromosomes_ntuple: list[tuple[Chromosome, ...] | Chromosome],
        callback: Callable[[tuple[Chromosome, ...]], str],
        k: int,
    ):
        candidate_prompts: list[str] = []
        replicated_ids: list[int] = []
        for chromosomes in chromosomes_ntuple:
            if isinstance(chromosomes, Chromosome):
                c_ids = chromosomes.id
                chromosomes = (chromosomes,)

            else:
                c_ids = tuple(c.id for c in chromosomes)

            replicated_ids += k * [c_ids]
            candidate_prompts += k * [callback(chromosomes)]

        return [
            self.ChromosomeObject(parent_id=c_ids, prompt=prompt, by=id(self.__class__))
            for c_ids, prompt in zip(
                replicated_ids,
                self.llm.generate_from_prompt(
                    prompts=candidate_prompts,
                    params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
                ),
            )
        ]

    def _mutate(self, chromosomes: list[Chromosome], k: int) -> list[Chromosome]:
        def mutate_callback(cs: tuple[Chromosome, ...]):
            return textwrap.dedent(
                f"""
                Consider the task to create similar sentences given the following sentence:

                {cs[0].prompt}

                Use the following as a context:

                {self.target}
            """
            )

        return self._generate_by_callback(chromosomes, mutate_callback, k=k)

    def _mixup(
        self, chromosomes: list[tuple[Chromosome, Chromosome]], k: int
    ) -> list[Chromosome]:
        # Based on: https://aclanthology.org/2023.emnlp-main.323.pdf and https://github.com/ServiceNow/promptmix-emnlp-2023/blob/c6f27bac6263e43a5b41dca01d3061edb88a4f35/src/generator/__init__.py#L87
        def mixup_callback(cs: tuple[Chromosome, ...]):
            return textwrap.dedent(
                f"""
                Consider the task of classifying between the following intents:

                {cs[0].prompt}
                {cs[1].prompt}

                Generate a diverse set of one short utterance where each utterance belongs to "{cs[0].prompt}" with the context {self.target}.
            """
            )

        return self._generate_by_callback(chromosomes, mixup_callback, k=k)
