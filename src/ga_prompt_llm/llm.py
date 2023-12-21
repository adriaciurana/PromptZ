import logging
import platform
import re
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from chromosome import Chromosome
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import openai
from dotenv import load_dotenv
from utils import AGGREGATE_STRINGS, Register, batch_processing


class LLM(ABC):
    IS_NAIVE = False

    def __init__(self, max_batch: int = 10, device: str = "cuda:0") -> None:
        self.max_batch = max_batch
        self.device = device if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        ...

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        return self.generate_from_prompt(prompts, params)


@Register("LLM")
class MockLLM(LLM):
    IS_NAIVE = True

    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(max_batch=max_batch, device=device)

    @batch_processing(AGGREGATE_STRINGS)
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        return [p for p in prompts]  # just copy :)

class HuggingFaceLLM(LLM):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Callable[[str | torch.device], PreTrainedModel],
        max_batch: int = 10,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 50,
            "num_beams": 2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
    ) -> None:
        super().__init__(max_batch, device)
        self._tokenizer = tokenizer
        self._model = model(device)

        self._default_params = default_params

    @batch_processing(AGGREGATE_STRINGS)
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        if params is None:
            params = {}
            params.update(self._default_params)

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

class OpenAIAPILLM(LLM):
    def __init__(
            self,
            max_batch: int = 10,
            device: str = "cuda:0",
            openai_model_id: str = "gpt-3.5-turbo-instruct",
            environ_variable: str = "OPENAI_API_KEY",
            default_params: dict[str, Any] = {
                "temperature": 0.8,
                "max_tokens": 500,
            },) -> None:
        super().__init__(max_batch, device)
        # Load environment variables.
        load_dotenv()
        # Set key.
        openai.api_key = os.getenv(environ_variable)
        # Set model id.
        self._openai_model_id = openai_model_id
        self._default_params = default_params

    def generate_from_prompt(self, prompts: list[str], params: dict[str, Any] | None = None) -> list[str]:
        if params is None:
            params = {}
            params.update(self._default_params)
        
        try:
            return [re.sub("\n", "", openai.completions.create(
                model=self._openai_model_id,
                prompt=prompt,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"]
            ).choices[0].text).strip("\"") for prompt in prompts]
        except:
            return [re.sub("\n", "", openai.completions.create(
                model=self._openai_model_id,
                prompt=prompt,
                temperature=0.8,
                max_tokens=500
            ).choices[0].text).strip("\"") for prompt in prompts]
    
    def __call__(self, population: list[Chromosome], params: dict[str, Any] | None = None) -> list[str]:
        return super().__call__(population, params)

@Register("LLM")
class Bloom(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 50,
            "num_beams": 2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("bigscience/bloom-560m"),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "bigscience/bloom-560m",
                torch_dtype="auto",
                device_map=device,
                load_in_4bit=True if platform.system() != "Windows" else False,
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )


@Register("LLM")
class Flan(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 50,
            "num_beams": 2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-small"),
            model=lambda device: AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-small",
                torch_dtype="auto",
                device_map=device,
                load_in_4bit=True if platform.system() != "Windows" else False,
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )


@Register("LLM")
class M0(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 50,
            # "num_beams": 2,
            # "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
    ) -> None:
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("bigscience/mt0-small"),
            model=lambda device: AutoModelForSeq2SeqLM.from_pretrained(
                "bigscience/mt0-small", torch_dtype="auto", device_map=device
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )


@Register("LLM")
class Phi2(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 2,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {"max_new_tokens": 50},
    ) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained("microsoft/phi-2"),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                device_map=device,
                load_in_4bit=True if platform.system() != "Windows" else False,
                quantization_config=bnb_config
                if platform.system() != "Windows"
                else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        outputs = [
            re.sub(r"^(.|\n)*Answer:", "", re.sub(re.escape(prompt), "", output))
            for prompt, output in zip(
                prompts, self.generate_from_prompt(prompts, params)
            )
        ]
        return outputs


@Register("LLM")
class Mistral(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 50,
            "num_beams": 2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
    ) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1"
            ),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                device_map=device,
                load_in_4bit=True if platform.system() != "Windows" else False,
                quantization_config=bnb_config
                if platform.system() != "Windows"
                else None,
                torch_dtype=torch.bfloat16,
                # device_map="auto",
                trust_remote_code=True,
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        outputs = [
            re.sub(prompt, "", output)
            for prompt, output in zip(
                prompts, self.generate_from_prompt(prompts, params)
            )
        ]
        return [re.sub(r"\?\nAnswer: ", "", output) for output in outputs]


@Register("LLM")
class Solar(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 10,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 500,
            "num_beams": 2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
        },
    ) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(
                "Upstage/SOLAR-10.7B-Instruct-v1.0"
            ),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "Upstage/SOLAR-10.7B-Instruct-v1.0",
                device_map=device,
                load_in_4bit=True if platform.system() != "Windows" else False,
                quantization_config=bnb_config
                if platform.system() != "Windows"
                else None,
                torch_dtype=torch.bfloat16,
                # device_map="auto",
                trust_remote_code=True,
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        outputs = [
            re.sub(r"^(.|\n)*Answer:", "", re.sub(re.escape(prompt), "", output))
            for prompt, output in zip(
                prompts, self.generate_from_prompt(prompts, params)
            )
        ]
        return outputs

    @batch_processing(AGGREGATE_STRINGS)
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [
            self._tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        return [
            re.sub(re.escape(prompt), "", output)
            for prompt, output in zip(
                prompts, super().generate_from_prompt(prompts, params)
            )
        ]


@Register("LLM")
class RudeWizardVicuna(HuggingFaceLLM):
    def __init__(
        self,
        max_batch: int = 2,
        device: str = "cuda:0",
        default_params: dict[str, Any] = {
            "max_new_tokens": 500,
            "num_beams": 2,
            "temperature": 0.7,
            "do_sample": True,
        },
    ) -> None:
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        logging.getLogger("transformers").addHandler(logging.NullHandler())
        super().__init__(
            tokenizer=AutoTokenizer.from_pretrained(
                "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
            ),
            model=lambda device: AutoModelForCausalLM.from_pretrained(
                "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ",
                device_map=device,
                trust_remote_code=True,
                revision="main",
            ),
            max_batch=max_batch,
            device=device,
            default_params=default_params,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        logging.getLogger("transformers").removeHandler(logging.NullHandler())

    def __call__(
        self, population: list[Chromosome], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [c.prompt for c in population]
        outputs = self.generate_from_prompt(prompts, params)
        return outputs

    @batch_processing(AGGREGATE_STRINGS)
    def generate_from_prompt(
        self, prompts: list[str], params: dict[str, Any] | None = None
    ) -> list[str]:
        prompts = [
            f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and rude answers to the user's questions. USER: {prompt} ASSISTANT:"
            for prompt in prompts
        ]
        return [
            re.sub(re.escape(prompt), "", output)
            for prompt, output in zip(
                prompts, super().generate_from_prompt(prompts, params)
            )
        ]


if __name__ == "__main__":
    llm = OpenAIAPILLM()
    keywords = ["hotel", "contract", "wife", "category", "dealer"]
    chromosome_prompts = [
        Chromosome(prompt="Give a bunch of curse words, be very rude and creative."),
    ]
    solutions = llm(chromosome_prompts)
    for c, solution in zip(chromosome_prompts, solutions):
        print("Prompt:")
        print(c.prompt)
        print("Solution:")
        print(solution)
        print()
