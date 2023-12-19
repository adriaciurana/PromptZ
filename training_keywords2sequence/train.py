from pathlib import Path

import datasets
import nltk
import numpy as np
import spacy
import torch
from datasets import load_dataset, load_from_disk
from torch import nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

"""
    DATASET CREATION
"""
ENCODER_MAX_LENGTH = 64  # demo
DECODER_MAX_LENGTH = 256
TRAIN_DATASET = Path(__file__).parent / "seq2seq_dataset/train_keywords.hf"
TEST_DATASET = Path(__file__).parent / "seq2seq_dataset/test_keywords.hf"

NLP = spacy.load("en_core_web_sm")


def keywords_extractor(example):
    global NLP
    keywords: list[str] = []
    for token in NLP(example["abstract"]):
        if token.pos_ in ["VERB", "ADJ", "NOUN"]:
            keywords.append(token.text)

    return {"keywords": keywords, "abstract": example["abstract"]}


def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    batch_keywords = [" ".join(keywords) for keywords in batch["keywords"]]
    source, target = batch_keywords, batch["abstract"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


if not TRAIN_DATASET.exists() or not TEST_DATASET.exists():
    dataset = load_dataset("ccdv/arxiv-summarization")

if not TRAIN_DATASET.exists():
    train_dataset = dataset["train"]
    train_clean_dataset = train_dataset.map(
        keywords_extractor, remove_columns=["article"]
    )
    train_clean_dataset.save_to_disk("seq2seq_dataset/train_keywords.hf")

if not TEST_DATASET.exists():
    test_dataset = dataset["test"]
    test_clean_dataset = test_dataset.map(
        keywords_extractor, remove_columns=["article"]
    )
    test_clean_dataset.save_to_disk("seq2seq_dataset/test_keywords.hf")

"""
    MODEL SURGERY
"""
model_name = "sshleifer/distilbart-xsum-12-3"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Remove PE input:
class UseLessModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._constant_zero = torch.tensor(0.0, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove the PE!
        return self._constant_zero


model.model.encoder.embed_positions = UseLessModule()
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""
DATASET PREPARATION
"""
train_clean_dataset = load_from_disk(TRAIN_DATASET)
train_clean_dataset = train_clean_dataset.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, ENCODER_MAX_LENGTH, DECODER_MAX_LENGTH
    ),
    batched=True,
    remove_columns=train_clean_dataset.column_names,
)

test_clean_dataset = load_from_disk(TEST_DATASET)
test_clean_dataset = test_clean_dataset.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, ENCODER_MAX_LENGTH, DECODER_MAX_LENGTH
    ),
    batched=True,
    remove_columns=test_clean_dataset.column_names,
)

"""
    PREPARE TRAINING
"""
training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4,  # demo
    per_device_eval_batch_size=4,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

nltk.download("punkt", quiet=True)

metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_clean_dataset,
    eval_dataset=test_clean_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
