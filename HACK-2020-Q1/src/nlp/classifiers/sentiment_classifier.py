from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import numpy as np


def map_score(score):
    if 0.1 < score < 0.5:
        return 'mildy_positive'
    elif 0.5 <= score:
        return 'positive'
    elif -0.1 > score > -0.5:
        return 'mildy_negative'
    elif -0.5 >= score:
        return 'negative'
    else:
        return neutral


def convert_scores_to_clases(scores):
    return [map_score(score) for score in scores]


class SentimentClassifier:
    """
    Generates sentiment scores for texts.
    """
    def __init__(self, tokenizer=None, sentiment_model=None, score_normaliser=lambda x: x):

        self.tokenizer = tokenizer or tokenize
        self.sentiment_model = sentiment_model or SentimentIntensityAnalyzer()
        self.score_normaliser = score_normaliser

    def _break_text_into_sents(self, text) -> list:
        """
        Converts string into list of strings using the class sentence tokenizer
        """
        return self.tokenizer.sent_tokenize(text)

    def _get_sentiment_score_for_sents(self, sents):
        return [
            self.sentiment_model.polarity_scores(sent).get('compound', 0.)
            for sent in sents
            ]

    def _aggregate_scores(self, scores):
        """
        Lazy use of numpy...
        """
        return np.mean(scores)

    def calculate_sentiment(self, texts, convert_to_sents=False, aggregate_scores=True):
        if convert_to_sents:
            texts = self._break_text_into_sents(texts)
        scores = self._get_sentiment_score_for_sents(texts)
        if aggregate_scores:
            scores = self._aggregate_scores(scores)
        scores = self.score_normaliser(scores)
        return scores


# hacky test
if __name__ == '__main__':
    test_sentences = ["VADER is smart, handsome, and funny.", "A really bad, horrible book."]
    test_text = ' '.join(test_sentences)

    classifier = SentimentClassifier()
    np.testing.assert_almost_equal(classifier.calculate_sentiment(test_sentences), 0.005249999999999977)
    np.testing.assert_almost_equal(classifier.calculate_sentiment(test_text, convert_to_sents=True), 0.005249999999999977)

    negative_sentences = ["It was OK", "A really bad, horrible book."]
    mapping_classifier = SentimentClassifier(score_normaliser=convert_scores_to_clases)
    assert mapping_classifier.calculate_sentiment(negative_sentences, aggregate_scores=False) == ['mildy_positive', 'negative']
