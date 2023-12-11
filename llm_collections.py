from abc import ABC, abstractclassmethod

import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

class LLM(ABC):
    
    def __init__(self, device) -> None:
        # Set device.
        self.set_device(device)
    
    def set_device(self, device = None) -> None:
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._deivce = device
    
    @abstractclassmethod
    def generate_output(self, prompt):
        ...

class Flan(LLM):
    
    def __init__(self, device = None) -> None:
        # Instantiate parent.
        super().__init__(device)
        
        # Model name.
        self._model_name = "google/flan-t5-large"
        
        # Tokenizer and model from huggingface.
        self._tokenizer = T5Tokenizer.from_pretrained(self._model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._model_name, device_map=self._device)
    
    def generate_output(self, prompt):
        # Get the input ID and put it to device.
        input_ids = self._tokenizer(
            prompt, return_tensors="pt")["input_ids"].to(self._device)
        
        # Get the output.
        outputs = self._model.generate(input_ids, max_new_tokens=1000)
        
        # Return decoded.
        return self._tokenizer.decode(outputs[0])