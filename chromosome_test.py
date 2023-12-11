import re
import copy
from dataclasses import dataclass

from llm_collections import LLM

@dataclass
class Chromosome:

    def __init__(self, keywords: list, prompt_generation_model: LLM):
        # Attributes.
        self._keywords = keywords
        self._prompts = []
        
        # Generate the prompts.
        self.generate_prompts(prompt_generation_model)
    
    def __str__(self):
        return "\n".join(prompt for prompt in self._prompts)
    
    def get_keywords(self):
        return copy.deepcopy(self._keywords)
    
    def get_prompts(self):
        return copy.deepcopy(self._prompts)
    
    def get_initial_prompt(self):
        # Concatenate the keywords into one string.
        concatenated = ", ".join(keyword for keyword in self._keywords)
        concatenated = re.sub("_", " ", concatenated)
        
        # Generate the prompt for the prompt generation model.
        initial_prompt = f"Generate a prompt with these words: {concatenated}."
        initial_prompt += " The prompt will be another input to an LLM."
        
        return initial_prompt
    
    def generate_prompts(
            self,
            prompt_generation_model: LLM,
            n_prompts: int = 5):
        # Get the output with the initial prompt.
        output = prompt_generation_model.generate_output(
            self.get_initial_prompt())
        output = self._clean_model_output(output)
        
        # Append to the list.
        self._prompts.append(output)
        
        if n_prompts > 1:
            for _ in range(n_prompts - 1):
                next_input = f"Rephrase these sentences with different words: {output}"
                output = prompt_generation_model.generate_output(next_input)
                output = self._clean_model_output(output)
                # Append to the list.
                self._prompts.append(output)
    
    def _clean_model_output(self, output):
        output = re.sub("<pad>", "", output)
        output = re.sub("</s>", "", output)
        return output.strip()