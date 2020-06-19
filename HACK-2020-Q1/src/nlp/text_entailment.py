"""
This code is to perform Natural Language Inference on BBC content.

We use the following libraries:
    - huggingface transforms
    - pytorch
"""
from transformers import AutoTokenizer, AutoModel, BartForSequenceClassification, BartTokenizer

import torch
from torch.nn import functional as F


def load_bart_model_tokenizer(model_name):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer


def load_model_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_premise_hypothesis_vectors(premise, hypothesis, tokenizer, model):
    # run inputs through model and mean-pool over the sequence
    # dimension to get sequence-level representations
    inputs = tokenizer.batch_encode_plus([premise, hypothesis],
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    premise_rep = output[0].mean(dim=1)
    hypothesis_reps = output[1].mean(dim=1)
    return premise_rep, hypothesis_reps


def get_entailment_label(prob):
    if prob <= 0.4:
        return 'contradiction'
    elif prob >= 0.6:
        return 'entailment'
    else:
        return 'neutral'


def get_premise_hypothesis_entailment(premise, hypothesis, tokenizer, model):
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:,1].item() * 100
    print(f'Probability that the label is true: {true_prob:0.2f}%')

model_name = 'facebook/bart-large-mnli'
model, tokenizer = load_bart_model_tokenizer(model_name)

premise = "Oriel College's governors vote to take down the statue of the Victorian colonialist Cecil Rhodes."
hypothesis = 'References the diamond trade'

get_premise_hypothesis_entailment(premise, hypothesis, tokenizer, model)
