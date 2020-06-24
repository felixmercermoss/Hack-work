import json

import requests


category_tags = {
    'isBLM': {'should': ['Black Lives Matter', 'BLM'],
              'should_not': []},
    'isBrexit': {'should':['Brexit', 'Operation Yellowhammer'],
                 'should_not': []},
    'isCovid': {'should': ['Covid', 'Coronavirus', 'Self-isolation', 'Lockdown', 'Contact tracing', 'Mers virus', 'Joint Biosecurity Centre (JBC)'],
                'should_not': []},
    'isEducation': {'should': ['Education'],
                    'should_not': []},
    'isImmigration': {'should': ['Immigration'],
                      'should_not': []},
    "isEconomy": {'should': ['Economy'],
                  'should_not': []},
}


football_teams = {
    'isArsenal': {'should': ['http://www.bbc.co.uk/things/c4285a9a-9865-2343-af3a-8653f7b70734#id']},
    'isBournemouth': {'should': ['http://www.bbc.co.uk/things/0280e88c-26bd-b24c-882f-ae0e5f4142f2#id']},
    'isBHA': {'should': ['http://www.bbc.co.uk/things/3d814d88-1f22-d042-a9a6-87fd9216a50a#id']},
    'isChelsea': {'should': ['http://www.bbc.co.uk/things/2acacd19-6609-1840-9c2b-b0820c50d281#id']},
    'isEverton': {'should': ['http://www.bbc.co.uk/things/48287aac-b015-1d4e-9c4e-9b8abac03789#id']},
    'isManUtd': {'should': ['http://www.bbc.co.uk/things/90d9a818-850b-b64f-9474-79e15a0355b8#id']},
    'isTottenham': {'should': ['http://www.bbc.co.uk/things/edc20628-9520-044d-92c3-fdec473457be#id']},
    'isCardiff': {'should': ['http://www.bbc.co.uk/things/bef8a4b5-1ad1-4347-b665-edf581ab7350#id']},
    'isFulham': {'should': ['http://www.bbc.co.uk/things/98c3db4b-498d-7c4b-acb5-d16ffb214f0d#id']},
    'isHuddersfield': {'should': ['http://www.bbc.co.uk/things/843257c2-309c-a749-9888-615bb448f887#id']},
    'isStoke': {'should': ['http://www.bbc.co.uk/things/ff3ad258-564a-3d46-967a-cefa4e65cfea#id']},
    'isSwansea': {'should': ['http://www.bbc.co.uk/things/98105d9f-b1db-0547-b8c7-581abf30c7e9#id']},
    'isWBA': {'should': ['http://www.bbc.co.uk/things/7f6edf4a-76f6-4b49-b16b-ff1e232eeb18#id']},
    'isHull': {'should': ['http://www.bbc.co.uk/things/10786d02-d602-084f-bfb5-3198a9bebfe7#id']},
    'isMiddlesbrough': {'should': ['http://www.bbc.co.uk/things/39a9af5a-c881-a74d-bae7-226ac220df03#id']},
    'isSunderland': {'should': ['http://www.bbc.co.uk/things/d5a95ba9-efe6-aa4e-afc4-9adc5f786e58#id']},
    'isAstonVilla': {'should': ['http://www.bbc.co.uk/things/9ce8f75f-4dc0-0f46-8e1b-742513527baf#id']},
    'isNewcastle': {'should': ['http://www.bbc.co.uk/things/34032412-5e2a-324d-bb3e-d0d4b16df2d4#id']},
    'isNorwich': {'should': ['http://www.bbc.co.uk/things/a700cc4d-72eb-a84d-8d7a-73ce435f6985#id']},
    'isBurnley': {'should': ['http://www.bbc.co.uk/things/279a3dc2-9195-264d-be3e-2a52bb61d4fe#id']},
    'isQPR': {'should': ['http://www.bbc.co.uk/things/81f6a64f-def0-0d40-bc3a-36dab5472f64#id']},
    'isWolverhampton': {'should': ['http://www.bbc.co.uk/things/b68a1520-32bc-eb42-8dc0-ccc1018e8e8f#id']},
    'isLeicester': {'should': ['http://www.bbc.co.uk/things/ff55aea0-83d7-834c-afc0-d21045f561e9#id']},
    'isWatford': {'should': ['http://www.bbc.co.uk/things/7d1d29cb-dab4-a24f-8600-393be1f354fe#id']},
    'isSouthampton': {'should': ['http://www.bbc.co.uk/things/6780f83f-a17a-e641-8ec8-226c285a5dbb#id']},
    'isSheffield': {'should': ['http://www.bbc.co.uk/things/75f90667-0306-e847-966e-085a11a8f195#id']},
    'isManCity': {'should': ['http://www.bbc.co.uk/things/4bdbf21d-d1ad-7147-ab08-612cd0dc20b4#id']},
}

starfruit_api = 'http://starfruit.virt.ch.bbc.co.uk'
mango_api = 'http://api.mango-en.virt.ch.bbc.co.uk'


def end_to_end_labelling(article, category_tags, tag_type='label', use_tags=True, use_headline=True, use_summary=True, use_body=True, use_starfish=True, use_mango=True):
    """
    Orchestration of boolean labelling for a single article and category.
    args:
        article (dict): article object to be labelled
        category_tags (dict): dictionary containing a list of "should" tags, and optional "should_not" tags
    Returns:
        True if text returned from various optional locations (tags, headline, summary, body, auto-taggers) contains at least one valid category tag,
        and no invalid tags.
    """
    text = []
    if use_tags:
        text += get_tags(article)
    if use_headline:
        text += [get_headline(article)]
    if use_summary:
        text += [get_summary(article)]
    if use_body:
        text += [get_body(article)]
    if use_starfish or use_mango:
        uri = get_uri(article)
        if use_starfish:
            response = get_results_from_autotagger(uri, api='http://starfruit.virt.ch.bbc.co.uk')
            text += parse_labels_from_starfruit_response(response, tag_type)
        if use_mango:
            response = get_results_from_autotagger(uri, api='http://api.mango-en.virt.ch.bbc.co.uk')
            text += parse_labels_from_mango_response(response, tag_type)
    return check_text_for_keywords(text, category_tags)


autotagger_cache = {}

def get_results_from_autotagger(asset_uri, api=starfruit_api):
    """
    Query starfruit or mango api with article URI to return auto-generated content tags.
    """
    uri = f'{api}/topics?uri=https://www.bbc.co.uk{asset_uri}'
    cached = autotagger_cache.setdefault(uri, None)
    if cached:
        return cached
    else:
        response = requests.get(uri)
        body = response.content
        decoded = json.loads(body.decode("utf-8"))
        autotagger_cache[uri] = decoded
        return decoded


def parse_labels_from_starfruit_response(response, return_type='label'):
    """
    Extract list of auto-generated labels from starfruit api response.
    """
    all_labels = response.get('results', [])
    if return_type == 'label':
        return [l.get(return_type, {}).get('en-gb', '') for l in all_labels]
    elif return_type == 'uri':
        return [l.get('@id', '') for l in all_labels]


def parse_labels_from_mango_response(response, return_type='label'):
    """
    Extract list of auto-generated labels from mango api response.
    """
    all_labels = response.get('results', [])
    if return_type == 'label':
        return [l.get(return_type, {}) for l in all_labels]
    elif return_type == 'uri':
        for l in all_labels:
            for uri in l.get('same_as', []):
                if uri.startswith("http://www.bbc.co.uk/things/"):
                    return uri + "#id"

        return ''

def get_tags(article):
    """
    Extract tags from within article json.
    """
    all_tags = article.get('metadata', {}).get('tags', {}).get('about', [])
    tag_names = [t.get('thingLabel') for t in all_tags]
    return tag_names


def get_headline(article):
    """
    Extract headline from within article json.
    """
    return article.get('promo', {}).get('headlines', {}).get('headline', '')


def get_summary(article):
    """
    Extract summary from within article json.
    """
    return article.get('promo', {}).get('summary', '')


def get_body(article):
    """
    Extract body text from within article json.
    """
    text = ''
    for t in article.get('content', {}).get('blocks', []):
        text += t.get('text', '')
    return text


def get_uri(article):
    """
    Extract uri from within article json.
    """
    return article.get('metadata', {}).get('locators', {}).get('assetUri', '')


def check_text_for_keywords(text, category_tags):
    """
    Function checks if text includes category tags, excluding invalid tags.
    Args:
        text (list): list of text segments to be searched (these may be tags, headlines, or body text)
        category_tags (dict): dictionary containing a list of "should" tags, and optional "should_not" tags
    Returns:
        True if text contains at least one valid category tag, and no invalid tags.
    """
    all_text = (' ').join([t.lower() for t in text])
    should = [t.lower() for t in category_tags.get('should', [])]
    should_not = [t.lower() for t in category_tags.get('should_not', [])]

    for s in should_not:
        if all_text.find(s) >= 0:
            return False
    for s in should:
        if all_text.find(s) >= 0:
            return True
    return False
