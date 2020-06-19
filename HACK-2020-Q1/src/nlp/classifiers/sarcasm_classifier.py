import pickle as pkl


from transformers import AutoTokenizer, AutoModel
import numpy as np

def load_model_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class SentenceBertVectorizer:

    def __init__(self, model_name=None, tokenizer=None, model=None):
        if model_name and (tokenizer is None or model is None):
            loaded_model, loaded_tokenizer = load_model_tokenizer(model_name)
        self.tokenizer = tokenizer or loaded_tokenizer
        self.model = model or loaded_model

    def vectorize_texts(self, texts, chunksize=None):
        if chunksize:
            print(f'Generating {round(len(texts) / chunksize)} chunks of size {chunksize} from {len(texts)} texts')
            texts = batch(texts, chunksize)
        else:
            texts = [texts]

        vectors = []
        for chunk in texts:
            inputs = self.tokenizer.batch_encode_plus(chunk,
                                                 return_tensors='pt',
                                                 pad_to_max_length=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            output = self.model(input_ids, attention_mask=attention_mask)[0]
            sentence_reps = output.mean(dim=1)
            vectors.extend(sentence_reps.detach().numpy())

        return np.array(vectors)



class SarcasmClassifier:

    def __init__(self, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer

    def load_classifier(self, classifier_fpath):
        """
        Load classifier object from pkl file
        """
        with open(classifier_fpath, 'rb') as fin:
            self.classifier = pkl.load(fin)
        print(f'Loaded classifier {self.classifier} from {classifier_fpath}')

    def predict(self, texts):
        vectors = self.vectorizer.vectorize_texts(texts)
        preds = self.classifier.predict(vectors)
        return preds


# more dodgy tests
if __name__ == '__main__':
    vectorizer = SentenceBertVectorizer(model_name='deepset/sentence_bert')
    sarcasm_classifier = SarcasmClassifier(None, vectorizer)
    sarcasm_classifier.load_classifier('sarcasm_clf.pkl')

    preds = sarcasm_classifier.predict(["Wow. You're about as cool as the sun", "Florida man found to be the most successful at hunting dinosaurs"])
    assert preds == [0, 1]
