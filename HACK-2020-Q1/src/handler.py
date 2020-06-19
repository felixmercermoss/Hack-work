import glob
import json
import os

from src.ares_enrichment import NewsAresItem, SportAresItem

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


def get_list_of_file_paths(PATH_TO_DATA):
    return glob.glob(PATH_TO_DATA + '/**/*', recursive=True)


def generate_enriched_file_path(file_path, write_dir):
    file_name = file_path.split('/')[-1]

    return os.path.join(write_dir, file_name)


def write_file(file_name, obj):
    with open(file_name, 'w') as fo:
        json.dump(obj, fo)

def enrich_raw_ares():
    """
    Loop through each article in data directory, then read data, enrich the data then write the data to a new file
    """
    write_dir = PATH_TO_DATA + "_feature_enriched"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    file_paths = get_list_of_file_paths(PATH_TO_DATA)

    for file_path in file_paths:
        with open(file_path, 'r') as fo:
            parsed_article = json.load(fo)

        enriched_article = NewsAresItem(ares_dict=parsed_article, mango_enrichment=True)
        datascapes_features = get_datascapes_features(enriched_article)
        parsed_article['datascapes_features'] = datascapes_features
        field_path_enriched = generate_enriched_file_path(file_path, write_dir)
        write_file(field_path_enriched, parsed_article)


def get_party_position(party, political_parties):
    position = [p.get('value') for p in political_parties if p.get('uri') == party.get('uri')][0]
    return position


def get_datascapes_features(enriched_article):
    datascapes_features = {}
    datascapes_features['politicalScore'] = str(enriched_article.political_score)
    datascapes_features['genderScore'] = str(enriched_article.female_proportion)
    party_score = []
    for party in enriched_article.mango_enricher.political_parties_mentioned:
        score = enriched_article.mango_enricher.political_party_refs.get(party.get('label', party.get('surface_form', '')), 0)
        position = get_party_position(party, enriched_article.mango_enricher.political_parties_referenced)
        p_score = {'name': party.get('label'),
                   'position': position,
                   'score': str(score)}
        party_score.append(p_score)

    return datascapes_features


if __name__ == '__main__':
    enrich_raw_ares()