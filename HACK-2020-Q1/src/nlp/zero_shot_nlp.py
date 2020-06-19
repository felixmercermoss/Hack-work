import numpy as np
from scipy.spatial.distance import cosine

from transformers import AutoTokenizer, AutoModel

import torch
from torch.nn import functional as F

from gensim.models import KeyedVectors

"""
Example of sentence labelling using zero-shot learning

Modified version of
https://joeddav.github.io/blog/2020/05/29/ZSL.html
"""


def get_sentence_vectors(texts, tokenizer, model):
    inputs = tokenizer.batch_encode_plus(texts,
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_rep = output.mean(dim=1)
    return sentence_rep


def get_sentence_label_vectors(sentence, label, tokenizer, model):
    # run inputs through model and mean-pool over the sequence
    # dimension to get sequence-level representations
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_rep = output[:1].mean(dim=1)
    label_reps = output[1:].mean(dim=1)
    return sentence_rep, label_reps

def top_labels(sentence, label, tokenizer, model, top_n=5, print_top=True, Z=None):
    sentence_rep, label_reps = get_sentence_label_vectors(sentence, label, tokenizer, model)
    # now find the labels with the highest cosine similarities to
    # the sentence

    if Z:
        sentence_projection = torch.matmul(sentence_rep, Z)
        label_projection = torch.matmul(label_reps, Z)
    else:
        sentence_projection = sentence_rep
        label_projection = label_reps

    # refactor the below and the compare_labels function
    similarities = F.cosine_similarity(sentence_projection, label_projection)
    closest = similarities.argsort(descending=True)#[:top_n]
    if print_top:
        for ind in closest:
            print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
    return closest, sentence_projection, label_projection

def compare_labels(labels, labels_projections):
    """
    Struggling to get this to work with torch cosine, trying scipy
    """
    print('Comparing Labels')

    labels_projections = labels_projections.numpy()
    for ind, projection in enumerate(labels_projections):
        print(f'Similarities for {labels[ind]}')
        similarities = [cosine(projection, inner_projection) for inner_projection in labels_projections]
        closest = similarities.argsort(descending=True)
        for inner_ind in closest:
            print(f'label: {labels[inner_ind]} \t similarity: {similarities[inner_ind]}')
    return closest


def torch_compare_labels(labels, labels_projections):
    """Torch cosine"""
    print('Comparing Labels')

    label_projections = label_projections.detach().numpy()
    previous_ind = 0
    for ind in range(len(labels_projections)):
        print(f'Similarities for {labels[ind]}')
        projection = labels_projections[previous_ind: ind + 1]
        similarities = F.cosine_similarity(projection, labels_projections) # might need to make this a tensor
        closest = similarities.argsort(descending=True)
        for inner_ind in closest:
            print(f'label: {labels[inner_ind]} \t similarity: {similarities[inner_ind]}')
        previous_ind = ind
    return closest

# an additional challenge noted by the author is we are using a sentence encoder to encode single or multi words.
# this creates a problem of saliency within the word embeddings.
# let us fit a least squares project from the sent embeddings to the word embeddings and apply this as a transform to our similarity function

def load_word2vec_model(model_fpath, binary_format=True):
    model = KeyedVectors.load_word2vec_format(model_fpath, binary=binary_format)
    return model


def get_top_k_vocab_counts(word2vec_model, k=10000):
    vocab_counts = {k: vocab_obj.count for k, vocab_obj in word2vec_model.vocab.items()}
    top_vocab = sorted(vocab_counts.items(), key=lambda item: item[1], reverse=True)[:k]
    vocab, counts = zip(*top_vocab)
    return vocab, counts

def fit_least_squares(tokenizer, sentence_model, word2vec_model, top_k_terms=100):
    """
    Huggingface solved this using

    1. Take the top KK most frequent words V in the vocabulary of a word2vec model
    2. Obtain embeddings for each word using word2vec, Φword(V)
    3. Obtain embeddings for each word using S-BERT, Φsent(V)
    4. Learn a least-squares linear projection matrix with L2 regularization from Φsent(V) to Φword(V)

    Apply this transformation for this
        c^=argc∈Cmax​cos(Φsent​(x)Z,Φsent​(c)Z)
    """

    # top K words
    top_k_words = word2vec_model.index2entity[:top_k_terms]

    # generate word embeddings for top k words
    word_embeddings = [word2vec_model[word] for word in top_k_words]
    # generate sentence embeddings for top k words
    sentence_embeddings = get_sentence_vectors(top_k_words, tokenizer, sentence_model)
    Z, qr = torch.lstsq(torch.from_numpy(np.array(word_embeddings)), sentence_embeddings)
    # fit least squares
    # Z, residuals, rank, s = np.linalg.lstsq(word_embeddings, sentence_embeddings.detach().numpy())
    return Z, qr #, residuals, rank, s


tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')

# let us print a couple of examples for this
sentence = 'Who are you voting for in 2020?'
labels = ['business', 'art & culture', 'politics']
print(sentence)
closest, sentence_projection, label_projection = top_labels(sentence, labels, tokenizer, model)

print('========')
# let us print a couple of examples for this
sentence = 'Cancer care in England has faced major disruption during the pandemic with big drops in numbers being seen following urgent referrals by GPs, figures show.'
labels = ['sad', 'neutral', 'happy']
print(sentence)
closest, sentence_projection, label_projection = top_labels(sentence, labels, tokenizer, model)

print('========')
sentence = 'Reni Eddo-Lodge has criticised the UK publishing industry after she became the first ever black British author to top the paperback non-fiction chart.'
labels = ['music', 'film', 'books']
print(sentence)
closest, sentence_projection, label_projection = top_labels(sentence, labels, tokenizer, model)


w2v_model = load_word2vec_model("GoogleNews-vectors-negative300.bin")
# this is to help incorporate salinecy into the label embeddings
Z, qr = fit_least_squares(tokenizer, model, w2v_model)
print(Z)
print('======== Post lstsq =========')

# let us print a couple of examples for this
sentence = 'Who are you voting for in 2020?'
labels = ['business', 'art & culture', 'politics']
print(sentence)
closest, sentence_projection, label_projection = top_labels(sentence, labels, tokenizer, model, Z)
compare_labels(labels, label_projection)
print('========')
# let us print a couple of examples for this
sentence = 'Cancer care in England has faced major disruption during the pandemic with big drops in numbers being seen following urgent referrals by GPs, figures show.'
labels = ['sad', 'neutral', 'happy']
print(sentence)
closest, sentence_projection, label_projection = top_labels(sentence, labels, tokenizer, model, Z)


print('========')
sentence = 'Reni Eddo-Lodge has criticised the UK publishing industry after she became the first ever black British author to top the paperback non-fiction chart.'
labels = ['music', 'film', 'books']
print(sentence)
closest, sentence_projection, label_projection = top_labels(sentence, labels, tokenizer, model, Z)
