import glob
import json
import os

from src.ares_enrichment import NewsAresItem, SportAresItem

PATH_TO_DATA = 'data/toy_dataset/'
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


def enrich_raw_ares():
    """
    Loop through each article in data directory, then read data, enrich the data then write the data to a new file
    """
    write_dir = PATH_TO_DATA + "_feature_enriched"
    os.mkdir(write_dir)
    file_paths = get_list_of_file_paths(PATH_TO_DATA)

    for file_path in file_paths:
        open(file_paths, 'r') as fo:
            parsed_article = json.load(fo)

        enriched_article = NewsAresItem(ares_dict=parsed_article)
        datascapes_features = {}
        datascapes_features['politicalScore'] = enriched_article.political_score
        datascapes_features['genderScore'] = enriched_article.female_proportion
        party_score = []
        for party in POLITICAL_PARTIES:
            score = enriched_article.mango_enricher.political_party_refs.get(party, 0)
            position =












if __name__ == '__main__':
    enrich_raw_ares()