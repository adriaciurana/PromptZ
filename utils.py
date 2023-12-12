import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

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