from abc import ABC, abstractclassmethod

import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

from chromosome import Chromosome


class LLM(ABC):
    def __init__(self, max_batch: int = 10, device: str = "cuda:0") -> None:
        self.max_batch = max_batch
        self.device = device

    @abstractclassmethod
    def tokenizer_population(self, population: list[Chromosome]) -> None:
        ...


class Bloom(LLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 50,
    ) -> None:
        super().__init__(max_batch, device)
        self.result_length = result_length
        self._tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")
        self._model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3").to(
            self.device
        )

    def tokenizer_population(self, population: list[Chromosome]) -> None:
        for c in population:
            c.tokens = self._tokenizer(c.prompt, return_tensors="pt")["input_ids"]

    def __call__(self, population: list[Chromosome]):
        batch_prompts = torch.cat(
            [c.tokens for c in population],
            dim=0,
        ).to(self.device)

        return self._tokenizer.decode(
            self._model.generate(batch_prompts, max_length=self.result_length)
        )
