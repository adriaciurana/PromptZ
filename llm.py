from abc import ABC, abstractclassmethod
from typing import Iterator

import torch
from chromosome import Chromosome
from torch.nn import functional as F

# from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LLM(ABC):
    def __init__(self, max_batch: int = 10, device: str = "cuda:0") -> None:
        self.max_batch = max_batch
        self.device = device if torch.cuda.is_available() else "cpu"

    @abstractclassmethod
    def tokenizer_population(self, population: list[Chromosome]) -> None:
        ...

    def pad_sequences(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        max_length = max(tokens, key=lambda t: t.shape[-1]).shape[-1]
        output_tensor = []
        for t in tokens:
            output_tensor.append(F.pad(t, (0, max_length - t.shape[-1]), "constant", 0))

        return torch.cat(output_tensor, dim=0)


class Bloom(LLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 50,
    ) -> None:
        super().__init__(max_batch, device)
        self.result_length = result_length
        self._tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-small")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            "bigscience/mt0-small",
            torch_dtype="auto",
            device_map=self.device,
            load_in_8bit=True,
        )

    def tokenizer_population(self, population: list[Chromosome]) -> None:
        for c in population:
            c.tokens = self._tokenizer.encode(
                c.prompt, return_tensors="pt"
            )  # ["input_ids"]

    def __call__(self, population: list[Chromosome]) -> list[str]:
        batch_prompts = self.pad_sequences([c.tokens for c in population]).to(
            self.device
        )

        return self._tokenizer.batch_decode(
            self._model.generate(batch_prompts, max_length=self.result_length),
            skip_special_tokens=True,
        )
