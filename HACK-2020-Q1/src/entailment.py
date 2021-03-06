import os
import sys

import numpy as np
import tensorflow as tf

from src.nlp.text_entailment import load_bart_model_tokenizer
from src.boolean_features import get_body
from src.nlp.text_entailment import get_premise_hypothesis_entailment
from src.timer import Timer

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

entailment_categories = {'isRacial': ['race', 'racism', 'blm', 'black lives matter'],
                         'isProtest': ['protest', 'boycott', 'demonstration', 'strike', 'riot'],
                         'isLawAndOrder': ['police', 'crime', 'prison', 'court', 'law enforcement', 'officer', 'trial', 'charged', ]}


#with Timer('Instantiating Roberta Q&A'):
#    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#    model = TFRobertaForQuestionAnswering.from_pretrained('roberta-base')
model_name = 'facebook/bart-large-mnli'
model, tokenizer = load_bart_model_tokenizer(model_name)


def get_premise_hypothesis_entailment_probability(premise, hypothesis, tokenizer, model):
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:,1].item()
    return true_prob

def get_label_from_entailment(article_text, category_tags, tokenizer=tokenizer, model=model, threshold=0.5, max_len=1024):
    '''
    Function applies a binary label to an article about whether it discusses the category, defined by a list of category keywords.
    Args:
        article_text (str):
        category_tags (list):
        threshold (float):
        tokenizer (BartTokenizer):
        model (BartForSequenceClassification):
    Returns:
        Boolean label indicating whether article belongs to category
    '''
    hypothesis = f'discusses {category_tags[0]}'
    for t in category_tags[1:]:
        hypothesis += f' or {t}'
    probability = get_premise_hypothesis_entailment_probability(article_text[:np.minimum(max_len, len(article_text))], hypothesis, tokenizer, model)
    return probability >= threshold
