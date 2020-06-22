import glob
import json
import os


from tqdm import tqdm
from src.boolean_features import football_teams, get_results_from_autotagger

PATH_TO_DATA = '/Users/mercef02/Dropbox (BBC)/DATASCAPES_DATA/news/ares_raw_20200618/news'


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

def save_mango(file_paths=None):
    """
    Loop through each article in data directory, then read data, enrich the data then write the data to a new file
    """
    write_dir = PATH_TO_DATA + "_mango"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    if file_paths is None:
        file_paths = get_list_of_file_paths(PATH_TO_DATA)

    with tqdm(range(len(file_paths))) as pbar:
        for file_path in file_paths:
            pbar.update(1)
            parsed_article = read_file(file_path)
            uri = parsed_article.get('metadata').get('locators').get('assetUri')
            response = get_results_from_autotagger(uri, api='http://api.mango-en.virt.ch.bbc.co.uk')
            field_path_enriched = generate_enriched_file_path(file_path, write_dir)
            write_file(field_path_enriched, response)


if __name__ == '__main__':
    save_mango()