import re
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, get_type_hints

import torch
from chromosome import Chromosome
from torch.nn import functional as F

# from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils import AGGREGATE_STRINGS, batch_processing


class LLM(ABC):
    def __init__(self, max_batch: int = 10, device: str = "cuda:0") -> None:
        self.max_batch = max_batch
        self.device = device if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        ...

    @abstractmethod
    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        ...


class HuggingFaceLLM(LLM):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Callable[[str | torch.device], PreTrainedModel],
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 100,
    ) -> None:
        super().__init__(max_batch, device)
        self._tokenizer = tokenizer
        self._model = model(device)
        self.result_length = result_length

    @batch_processing(AGGREGATE_STRINGS)
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        if params is None:
            params = {
                "max_length": self.result_length,
                "num_beams": 2,
                "no_repeat_ngram_size": 2,
                "early_stopping": True,
            }

        kwargs = {}
        kwargs.update(params)
        batch_tokens = self._tokenizer(prompts, return_tensors="pt", padding=True)
        batch_tokens["input_ids"] = batch_tokens["input_ids"].to(self.device)
        batch_tokens["attention_mask"] = batch_tokens["attention_mask"].to(self.device)
        kwargs.update(batch_tokens)
        with torch.no_grad():
            return self._tokenizer.batch_decode(
                self._model.generate(**kwargs),
                skip_special_tokens=True,
            )

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        return self.generate_from_prompt(prompts, params)


class Bloom(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 50,
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("bigscience/bloom-560m"),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "bigscience/bloom-560m",
                torch_dtype="auto",
                device_map=device,
                load_in_4bit=True,
            ),
            max_batch=max_batch,
            device=device,
            result_length=result_length,
        )


class Flan(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 50,
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-small"),
            model=lambda device: AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-small",
                torch_dtype="auto",
                device_map=device,
                load_in_4bit=True,
            ),
            max_batch=max_batch,
            device=device,
            result_length=result_length,
        )


class M0(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 50,
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("bigscience/mt0-small"),
            model=lambda device: AutoModelForSeq2SeqLM.from_pretrained(
                "bigscience/mt0-small",
                torch_dtype="auto",
                device_map=device
            ),
            max_batch=max_batch,
            device=device,
            result_length=result_length,
        )
        
class Mistral(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        result_length: int = 50,
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                torch_dtype="auto",
                device_map=device
            ),
            max_batch=max_batch,
            device=device,
            result_length=result_length,
        )
    
    @batch_processing(AGGREGATE_STRINGS)
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        if params is None:
            params = {
                "max_length": self.result_length,
                "num_beams": 2,
                "no_repeat_ngram_size": 2,
                "early_stopping": True,
            }

        kwargs = {}
        kwargs.update(params)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        batch_tokens = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        batch_tokens["input_ids"] = batch_tokens["input_ids"].to(self.device)
        batch_tokens["attention_mask"] = batch_tokens["attention_mask"].to(self.device)
        kwargs.update(batch_tokens)
        with torch.no_grad():
            return self._tokenizer.batch_decode(
                self._model.generate(**kwargs),
                skip_special_tokens=True,
            )

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        outputs = [re.sub(prompt, "", output) for prompt, output in zip(prompts, self.generate_from_prompt(prompts, params))]
        return [re.sub("\?\nAnswer: ", "", output) for output in outputs]

if __name__ == "__main__":
    llm = Mistral()
    chromosome_prompts = [
        Chromosome(
            prompt="Please answer the following question: Who was the president of USA?"
        ),
        Chromosome(
            prompt="Please answer the following question: Who was the president of USA?"
        ),
        Chromosome(
            prompt="Please answer the following question: Who was the president of USA?"
        ),
        Chromosome(
            prompt="Please answer the following question: Who was the president of USA?"
        ),
    ]
    solutions = llm(chromosome_prompts)
    print(solutions)
