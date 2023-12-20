import textwrap
import weakref
from abc import ABC, abstractmethod
from copy import copy
from random import choice
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
import torch
from chromosome import Chromosome, FixedLengthChromosome
from llm import LLM

if TYPE_CHECKING:
    from generator import Generator

NLP_SPACY = None
NLP_SPACY_LOOKUPS = None


def generate_by_callback(
    ChromosomeObject: Callable,
    llm: LLM,
    chromosomes_ntuple: list[tuple[Chromosome, ...] | Chromosome],
    callback: Callable[[tuple[Chromosome, ...]], str],
    by: int,
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
        candidate_prompts += k * callback(chromosomes)

    return [
        ChromosomeObject(parent_id=c_ids, prompt=prompt, by=by)
        for c_ids, prompt in zip(
            replicated_ids,
            llm.generate_from_prompt(
                prompts=candidate_prompts,
                params={"max_new_tokens": 40, "do_sample": True, "top_k": 50},
            ),
        )
    ]


def INIT_SPACY():
    global NLP_SPACY, NLP_SPACY_LOOKUPS
    try:
        import spacy
        from spacy.lookups import load_lookups
    except ImportError:
        print("Please install gensim if you want to use `Noise` mutator.")

    if NLP_SPACY is None:
        NLP_SPACY = spacy.load("en_core_web_lg")
        NLP_SPACY_LOOKUPS = load_lookups("en", ["lexeme_prob"])
        NLP_SPACY.vocab.lookups.add_table(
            "lexeme_prob", NLP_SPACY_LOOKUPS.get_table("lexeme_prob")
        )


class MutatorModule:
    def __init__(self):
        self._generator_ref: "Generator" | None = None

    def register(self, generator: "Generator") -> None:
        self._generator_ref = weakref.ref(generator)

    @property
    def _generator(self) -> "Generator":
        return self._generator_ref()


class CrossOver(ABC, MutatorModule):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self,
        chromosome_a: Chromosome,
        chromosome_b: Chromosome,
        by: int,
    ) -> Chromosome:
        ...


class FixedLengthMixSentences(CrossOver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        chromosome_a: FixedLengthChromosome,
        chromosome_b: FixedLengthChromosome,
        by: int,
    ) -> FixedLengthChromosome:
        assert isinstance(chromosome_a, FixedLengthChromosome) and isinstance(
            chromosome_b, FixedLengthChromosome
        )
        words_a = chromosome_a.prompt.split(" ")
        words_b = chromosome_b.prompt.split(" ")

        tokens_mixed = []
        for word_idx, (t_a, t_b) in enumerate(zip(words_a, words_b)):
            if (
                chromosome_a.mutable_mask is not None
                and not chromosome_a.mutable_mask[word_idx]
            ):
                continue

            if (
                chromosome_b.mutable_mask is not None
                and not chromosome_b.mutable_mask[word_idx]
            ):
                continue

            if np.random.rand() < 0.5:
                tokens_mixed.append(t_a)
            else:
                tokens_mixed.append(t_b)

        tokens_mixed = torch.stack(tokens_mixed, dim=0)

        return FixedLengthChromosome(
            parent_id=(chromosome_a.parent_id, chromosome_b.parent_id),
            tokens=tokens_mixed,
            mutable_mask=chromosome_a.mutable_mask,
            by=by,
        )


class LLMCrossOver(CrossOver):
    def __init__(self) -> None:
        super().__init__()

    def _crossover_callback(self, cs: tuple[Chromosome, ...]):
        return textwrap.dedent(
            f"""
            Consider the task of classifying between the following intents:
            
            {cs[0].prompt}
            {cs[1].prompt}

            Generate a diverse set of one short utterance where each utterance belongs to "{cs[0].prompt}" with the context {self._generator.target}.
        """
        )

    def __call__(
        self,
        chromosome_a: Chromosome,
        chromosome_b: Chromosome,
        by: int,
    ) -> Chromosome:
        return generate_by_callback(
            self._generator.ChromosomeObject,
            self._generator.llm,
            [(chromosome_a, chromosome_b)],
            self._crossover_callback,
            by,
            k=1,
        )[0]


class Mutator(ABC, MutatorModule):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, chromosome: Chromosome, by: int) -> Chromosome:
        ...


class FixedLengthNoise(Mutator):
    def __init__(self, k: int = 10) -> None:
        super().__init__()
        INIT_SPACY()
        self.k = k

    def _most_similars(self, word: str, topk: int = 5) -> Iterator[tuple[str, float]]:
        global NLP_SPACY
        word_obj = NLP_SPACY.vocab[word]
        queries = [
            w
            for w in word_obj.vocab
            if w.is_lower == word_obj.is_lower
            and w.prob >= -15
            and np.count_nonzero(w.vector)
        ]

        by_similarity = sorted(
            queries, key=lambda w: word_obj.similarity(w), reverse=True
        )
        return (
            (w.lower_, w.similarity(word_obj))
            for w in by_similarity[: topk + 1]
            if w.lower_ != word_obj.lower_
        )

    def __call__(
        self, chromosome: FixedLengthChromosome, by: int
    ) -> FixedLengthChromosome:
        words = copy(chromosome.prompt)
        rand_prob = torch.rand(*words.shape) > 0.5
        for word_idx, (word, is_mutable) in enumerate(zip(words, rand_prob)):
            if not is_mutable or (
                chromosome.mutable_mask is not None
                and not chromosome.mutable_mask[word_idx]
            ):
                continue

            similar_words = self._most_similars(word, topk=self.k)
            p = np.array([s[1] for s in similar_words])
            p /= p.sum()
            word_sim_idx = np.random.choice(
                np.arange(len(similar_words)), replace=False, p=p
            )
            words[word_idx] = similar_words[word_sim_idx]

        return FixedLengthChromosome(
            parent_id=chromosome.id,
            tokens=words,
            mutable_mask=chromosome.mutable_mask,
            by=by,
        )


class LLMMutator(Mutator):
    def __init__(self) -> None:
        super().__init__()

    def _mutate_callback(self, cs: tuple[Chromosome, ...]):
        return textwrap.dedent(
            f"""
            Using the following text prompt:
            
            {cs[0].prompt}

            Create a similar prompt that can be better if you have to answer the following text:
            
            {self._generator.target}
        """
        )

    def __call__(
        self,
        chromosome: Chromosome,
        by: int,
    ) -> Chromosome:
        return generate_by_callback(
            self._generator.ChromosomeObject,
            self._generator.llm,
            [chromosome],
            self._mutate_callback,
            by,
            k=1,
        )[0]


class VariationsPolicy:
    def __init__(
        self,
        crossovers: list[CrossOver],
        mutators: list[Mutator],
        prob_to_crossover: float = 0.1,
        prob_to_mutate: float = 0.001,
    ) -> None:
        self._mutators = mutators
        self._crossovers = crossovers
        self._prob_to_crossover = prob_to_crossover
        self._prob_to_mutate = prob_to_mutate

        self._generator_ref: "Generator" | None = None

    def register(self, generator: "Generator") -> None:
        self._generator_ref = weakref.ref(generator)
        for m in self._mutators + self._crossovers:
            m.register(generator)

    @property
    def _generator(self) -> "Generator":
        return self._generator_ref()

    def init(
        self, population: list[Chromosome], k: int, by: int
    ) -> Iterator[Chromosome]:
        for c in population:
            for _ in range(k):
                mutator_method = choice(self._mutators)
                c = mutator_method(c, by=by)

                yield c

        yield from ()

    def __call__(
        self,
        pair_parents: Iterator[tuple[Chromosome, Chromosome]],
        by: int,
    ) -> Iterator[Chromosome]:
        for c_a, c_b in pair_parents:
            if np.random.rand() < self._prob_to_crossover:
                crossover_method = choice(self._crossovers)
                c_a = crossover_method(c_a, c_b, by=by)
                c_b = crossover_method(c_b, c_a, by=by)

            if np.random.rand() < self._prob_to_mutate:
                mutator_method = choice(self._mutators)
                c_a = mutator_method(c_a, by=by)

            if np.random.rand() < self._prob_to_mutate:
                mutator_method = choice(self._mutators)
                c_b = mutator_method(c_b, by=by)

            yield c_a
            yield c_b

        yield from ()
