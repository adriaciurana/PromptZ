from random import sample, shuffle

import spacy
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLP = spacy.load("en_core_web_sm")

model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-49000")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-3")


class UseLessModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._constant_zero = torch.tensor(0.0, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove the PE!
        return self._constant_zero


model.model.encoder.embed_positions = UseLessModule()


def keywords_extractor(text):
    global NLP
    keywords: list[str] = []
    for token in NLP(text):
        if token.pos_ in ["VERB", "ADJ", "NOUN"]:
            keywords.append(token.text)

    return keywords


def run(kws: list[str]):
    batch = tokenizer(" ".join(kws), return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"])
    print("Keywords:", kws)
    print("Sequence:", tokenizer.batch_decode(generated_ids, skip_special_tokens=True))


text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
keywords = keywords_extractor(text)
keywords_original = [k for k in keywords]

# keywords_subset = keywords

run(keywords)
shuffle(keywords)
run(keywords)
shuffle(keywords)
run(keywords)
