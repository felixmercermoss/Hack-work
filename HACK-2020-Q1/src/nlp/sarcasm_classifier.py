"""
File to train a classifier to detect sarcasm in text
"""


from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import *

import pandas as pd

import json
import numpy as np

import tqdm
from collections import Counter


data_path = '/Users/harlaa04/workspace/Hack-work/HACK-2020-Q1/data/sarcasm/Sarcasm_Headlines_Dataset.json'


def read_ndjson_data(path):
    with open(path) as fin:
        for line in fin:
            yield json.loads(line)


def load_model_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generate_sentence_embeddings(texts, tokenizer, model, chunksize=None):
    if chunksize:
        print(f'Generating {round(len(texts) / chunksize)} chunks of size {chunksize} from {len(texts)} texts')
        texts = batch(texts, chunksize)
    else:
        texts = [texts]
    vectors = []
    for chunk in tqdm.tqdm(texts):
        inputs = tokenizer.batch_encode_plus(chunk,
                                             return_tensors='pt',
                                             pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = model(input_ids, attention_mask=attention_mask)[0]
        sentence_reps = output.mean(dim=1)
        vectors.extend(sentence_reps.detach().numpy())

    return np.array(vectors)


data = pd.DataFrame(read_ndjson_data(data_path))

print(data.head())
headline = data['headline']
is_sarcastic = data['is_sarcastic']

X_train, X_test, y_train, y_test = train_test_split(headline, is_sarcastic, test_size=0.33)


model_name = 'deepset/sentence_bert'

model, tokenizer = load_model_tokenizer(model_name)

print(len(X_train))
n_samples = 10000
sampled_y_train = y_train[:n_samples]


class_dist = Counter(sampled_y_train)
print(f'train class_dist {class_dist}')

X_train_vec = generate_sentence_embeddings(X_train[:n_samples], tokenizer, model, chunksize=50)
print(X_train_vec.shape)

X_test_vec = generate_sentence_embeddings(X_test[:n_samples], tokenizer, model, chunksize=50)
sampled_y_test = y_test[:n_samples]

class_dist = Counter(sampled_y_test)
print(f'test class_dist {class_dist}')


clf = SVC(kernel='linear')

clf.fit(X_train_vec, sampled_y_train)

y_pred = clf.predict(X_train_vec)


# Also need to consider recall.
print(f'Train accuracy: {accuracy_score(sampled_y_train, y_pred)}')

y_test_pred = clf.predict(X_test_vec)
print(f'Test accuracy: {accuracy_score(sampled_y_test, y_test_pred)}')


print("Let's perform a wee bit of grid search for the kernel")
# TODO very quick and dirty version of gridsearch + cv. Need more samples and proper sampling
hp = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': np.logspace(-2, 3, 5)
}

hp_opt_clf = SVC()

grid = GridSearchCV(estimator=hp_opt_clf, param_grid=hp)
grid.fit(X_train_vec, sampled_y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

print("Evaluating best estimator")
best_estimator = grid.best_estimator_

best_estimator.fit(X_train_vec, sampled_y_train)

y_pred = best_estimator.predict(X_train_vec)

print(f'Train accuracy: {accuracy_score(sampled_y_train, y_pred)}')

y_test_pred = best_estimator.predict(X_test_vec)
print(f'Test accuracy: {accuracy_score(sampled_y_test, y_test_pred)}')
