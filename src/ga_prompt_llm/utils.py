import functools
import logging
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT_PATH = Path(__file__).parent / "../../"


import torch

if TYPE_CHECKING:
    from llm import LLM


class BatchProcessing:
    def __init__(self, aggregation_function: Callable) -> None:
        self._aggregation_function = aggregation_function

    def __call__(self, f: Callable) -> Callable:
        hints = get_type_hints(f)
        first_arg = list(hints.keys())[0]

        @functools.wraps(wrapped=f)
        def wrapper(instance: "LLM", *args: list[Any], **kwargs: dict[str, Any]):
            output: list[Any] = []
            max_batch: int = instance.max_batch

            new_args = list(args)
            new_kwargs = {}
            new_kwargs.update(kwargs)

            if len(args) >= 1:
                x = args[0]
                del new_args[0]

            else:
                x = kwargs[first_arg]
                del new_kwargs[first_arg]

            for start_idx in range(0, len(x), max_batch):
                end_idx = min(len(x), start_idx + max_batch)
                batch_x = x[start_idx:end_idx]
                output.append(f(instance, batch_x, *new_args, **new_kwargs))

            return self._aggregation_function(output)

        return wrapper


batch_processing = BatchProcessing


def AGGREGATE_STRINGS(output: list[list[str]]) -> list[str]:
    return [seq for batch_seq in output for seq in batch_seq]


def AGGREGATE_TENSORS(output: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(output, dim=0)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


class Register:
    REGISTER: dict[str, dict[str, Any]] = defaultdict(dict)

    @classmethod
    def get(cls, name: str, key: str) -> dict[str, Any]:
        return cls.REGISTER[name][key]

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, module: Any) -> Callable:
        self.REGISTER[self._name][module.__name__] = module
        return module


class CachedByTime(dict):
    def __init__(self, interval: int = 5 * 60, max_time: int = 5 * 60):
        super().__init__()
        self._last_use: dict[str, int] = {}
        self.interval = interval
        self.max_time = max_time

        self._thread = Thread(target=self._check_cache, daemon=True)
        self._thread.start()

    def __getitem__(self, __key: Any) -> Any:
        self._last_use[__key] = time.time()
        return super().__getitem__(__key)

    def __setitem__(self, __key: Any, __value: Any) -> None:
        self._last_use[__key] = time.time()
        return super().__setitem__(__key, __value)

    def _check_cache(self):
        while True:
            current_time: int = time.time()
            to_remove: list[str] = []
            for key in self:
                if (current_time - self._last_use[key]) > self.max_time:
                    to_remove.append(key)

            for key in to_remove:
                del self[key]

            end_time = time.time()

            elapsed_time = end_time - current_time
            time.sleep(max(0, self.interval - elapsed_time))


class CacheWithRegister(CachedByTime):
    def __init__(
        self,
        registry_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] = {},
        interval: int = 5 * 60,
        max_time: int = 5 * 60,
    ) -> None:
        super().__init__(interval, max_time)
        self._registry_name = registry_name
        self._args = args
        self._kwargs = kwargs

        super().__init__()

    def __getitem__(self, __key: Any) -> Any:
        try:
            return super().__getitem__(__key)

        except KeyError:
            value = Register.get(self._registry_name, __key)(
                *self._args, **self._kwargs
            )
            self[__key] = value

        return value


class Keyword2SequenceModel:
    def __init__(
        self,
        model_folder_path: Path = ROOT_PATH / "keywords2sequence",
        device: str = "cuda:0",
        max_batch: int = 10,
    ) -> None:
        if not Path(model_folder_path).exists():
            raise FileNotFoundError(
                f"The model is not found, please download from here: https://drive.google.com/file/d/12mmVM2zX54f9HErMK9wRjh6iVwUfXmHC/view?usp=sharing and unzip in the root folder: {model_folder_path.parent}"
            )

        self.device = device
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_folder_path, device_map=self.device
        )
        self._model.model.encoder.embed_positions = UseLessModule()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-xsum-12-3"
        )
        self.max_batch = max_batch

    @batch_processing(AGGREGATE_STRINGS)
    def __call__(self, keywords: list[list[str]]) -> list[str]:
        batch = self._tokenizer(" ".join(keywords), return_tensors="pt")
        with torch.no_grad():
            generated_ids = self._model.generate(batch["input_ids"].to(self.device))
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
