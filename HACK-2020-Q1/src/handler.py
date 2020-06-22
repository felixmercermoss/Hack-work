import glob
import json
import os

import numpy as np
import pytest

from tqdm import tqdm
from src.boolean_features import get_body, football_teams

from src.ares_enrichment import NewsAresItem, SportAresItem
from src.boolean_features import end_to_end_labelling, category_tags
from src.entailment import get_label_from_entailment, entailment_categories
from src.nlp.classifiers.sarcasm_classifier import SentenceBertVectorizer, SarcasmClassifier
from src.nlp.classifiers.sentiment_classifier import SentimentClassifier, convert_scores_to_clases

PATH_TO_DATA = '/Users/mercef02/Projects/Hack-work/HACK-2020-Q1/data/toy_dataset'
POLITICAL_PARTIES = ['labour_party',
                     'conservative_party',
                     'liberal_democrats',
                     'scottish_national_party',
                     'democratic_unionist_party',
                     'sinn_féin',
                     'plaid_cymru',
                     'green_party',
                     'social_democratic_unionists',
                     'alliance_party_of_northern_ireland',
                     'brexit_party',
                     'uk_independence_party']

BOOLEAN_FEATURES = ['isBLM',
                    'isBrexit',
                    'isCovid',
                    'isEducation',
                    'isImmigration',
                    'isEconomy'
                    ]

ENTAILMENT_BOOLEAN_FEATURES = ['isRacial',
                               'isProtest',
                               'isLawAndOrder']


mapping_classifier = SentimentClassifier(score_normaliser=convert_scores_to_clases)
vectorizer = SentenceBertVectorizer(model_name='deepset/sentence_bert')
sarcasm_classifier = SarcasmClassifier(None, vectorizer)
sarcasm_classifier.load_classifier('nlp/classifiers/sarcasm_clf.pkl')


def get_list_of_file_paths(PATH_TO_DATA):
    return glob.glob(PATH_TO_DATA + '/**/*', recursive=True)


def generate_enriched_file_path(file_path, write_dir):
    file_name = file_path.split('/')[-1]

    return os.path.join(write_dir, file_name)


def write_file(file_name, obj):
    with open(file_name, 'w') as fo:
        json.dump(obj, fo)


def read_file(file_path):
    with open(file_path, 'r') as fo:
        parsed_article = json.load(fo)
    return parsed_article

def enrich_raw_ares(file_paths=None):
    """
    Loop through each article in data directory, then read data, enrich the data then write the data to a new file
    """
    write_dir = PATH_TO_DATA + "_feature_enriched"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    if file_paths is None:
        file_paths = get_list_of_file_paths(PATH_TO_DATA)

    with tqdm(range(len(file_paths))) as pbar:
        for file_path in file_paths[:1]:
            pbar.update(1)
            parsed_article = read_file(file_path)
            datascapes_features = get_datascapes_features(parsed_article)
            parsed_article['datascapes_features'] = datascapes_features
            field_path_enriched = generate_enriched_file_path(file_path, write_dir)
            write_file(field_path_enriched, parsed_article)


def get_party_position(party, political_parties):
    position = [p.get('value') for p in political_parties if p.get('uri') == party.get('uri')][0]
    return position


def decorate_article_with_score_features(enriched_article, datascapes_features):
    datascapes_features['politicalScore'] = str(enriched_article.political_score)
    datascapes_features['genderScore'] = str(enriched_article.female_proportion)
    party_score = []
    for party in enriched_article.mango_enricher.political_parties_mentioned:
        score = enriched_article.mango_enricher.political_party_refs.get(
            party.get('label', party.get('surface_form', '')), 0)
        position = get_party_position(party, enriched_article.mango_enricher.political_parties_referenced)
        p_score = {'name': party.get('label'),
                   'position': position,
                   'score': str(score)}
        party_score.append(p_score)

    datascapes_features['partyScore'] = party_score
    return datascapes_features


def decorate_article_with_boolean_features(parsed_article, datascapes_features):
    for category in BOOLEAN_FEATURES:
        datascapes_features[category] = end_to_end_labelling(parsed_article, category_tags[category])
    return datascapes_features


def decorate_article_with_entailment_boolean_features(parsed_article, datascapes_features):
    text = get_body(parsed_article)
    for category in ENTAILMENT_BOOLEAN_FEATURES:
        datascapes_features[category] = get_label_from_entailment(text, entailment_categories[category])
    return datascapes_features


def decorate_article_with_tonality_features(parsed_article, datascapes_features):
    datascapes_features['sentimentLabel'] = mapping_classifier.calculate_sentiment(
        parsed_article.combined_body_summary_headline, convert_to_sents=True)
    return datascapes_features


def decorate_article_with_sport_features(parsed_article, datascapes_features):
    team_features = []
    for team in football_teams:
        team_features.append(end_to_end_labelling(parsed_article, football_teams[team], tag_type='uri', use_body=False, use_summary=False, use_tags=False, use_headline=False))

    datascapes_features['premierLeagueTeams'] = team_features
    return datascapes_features


def decorate_article_with_sarcasm_features(parsed_article, datascapes_features):
    datascapes_features['sarcasmLabel'] = np.mean(sarcasm_classifier.predict(parsed_article.combined_body_summary_headline))
    return datascapes_features


def get_datascapes_features(parsed_article):
    enriched_article = NewsAresItem(ares_dict=parsed_article, mango_enrichment=True)
    datascapes_features = {}
    datascapes_features = decorate_article_with_score_features(enriched_article, datascapes_features)
    datascapes_features = decorate_article_with_boolean_features(parsed_article, datascapes_features)
    datascapes_features = decorate_article_with_entailment_boolean_features(parsed_article, datascapes_features)
    datascapes_features = decorate_article_with_tonality_features(enriched_article, datascapes_features)
    datascapes_features = decorate_article_with_sarcasm_features(enriched_article, datascapes_features)
    datascapes_features = decorate_article_with_sport_features(parsed_article, datascapes_features)

    return datascapes_features


if __name__ == '__main__':
    enrich_raw_ares()


def test_enrich_raw_ares():
    enrich_raw_ares(file_paths=['/Users/mercef02/Projects/Hack-work/HACK-2020-Q1/data/toy_dataset/14295731'])