import os
import re
from datetime import datetime

import pytz

# Datascapes-dev local imports
from mapper.dict_utils import convert_dict_underscore_keys_to_camelcase, remove_empty_elements
from mapper.mango import enrichedMangoTags, POLITICAL_PARTIES
from processing.html_cleaning_tools import clean_html_markup
from utils.utils import load_config_file

# news-wsoj-english-content-similarity-vectorisation-service local imports
# from html_cleaning_tools import clean_html_markup
# from dict_utils import convert_dict_underscore_keys_to_camelcase, remove_empty_elements
# from utils import load_config_file


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


class EnrichedAresItem:
    def __init__(self, ares_dict, mango_enrichment=False):
        """Maps and enriches the input Ares json to elasticsearch schema.
        Empty fields such as None, {}, [], are removed from the output to avoid elasticsearch 'null' values
        Input example: https://ares-broker.api.bbci.co.uk/api/asset/sport/tennis/42532950

        Args:
            ares_dict: For example,
            {
                "metadata": {
                    "locators":{"assetUri":"/sport/football/12570486"},
                    "lastPublished":1353921092,
                    "tags":{
                         "about":[{
                               "thingLabel":"Tennis",
                               "thingUri":"http://www.bbc.co.uk/things/0987d977-e389-964b-bbb0-0390f4c7a899#id",
                               "thingId":"0987d977-e389-964b-bbb0-0390f4c7a899"}]
                        }
            }

        Returns:
            es_dict: For example,
            {
                "assetUri": "/sport/football/12570486",
                "tags": [{"id": "0987d977-e389-964b-bbb0-0390f4c7a899", "label": "Tennis"}],
                "lastPublished": "2018-01-01T13:29:54+00:00"
            }
        """
        self.data = ares_dict
        if mango_enrichment:
            self.mango_enricher = enrichedMangoTags(get_value_or_none_from_asset(ares_dict, ['metadata',
                                                                                             'locators', 'assetUri']))
        else:
            self.mango_enricher = None

    @property
    def female_proportion(self):
        if self.mango_enricher:
            return self.mango_enricher.female_mentions_proportion
        else:
            return None

    @property
    def num_people_mentioned(self):
        if self.mango_enricher:
            if self.mango_enricher.people_mentioned:
                return len(self.mango_enricher.people_mentioned)
            else:
                return 0
        else:
            return 0

    @property
    def num_political_references(self):
        if self.mango_enricher:
            if self.mango_enricher.political_parties_referenced:
                return len(self.mango_enricher.political_parties_referenced)
            else:
                return 0
        else:
            return 0

    @property
    def num_politicians(self):
        if self.mango_enricher:
            if self.mango_enricher.politicians_mentioned:
                return len(self.mango_enricher.politicians_mentioned)
            else:
                return 0
        else:
            return 0

    @property
    def political_score(self):
        if self.mango_enricher:
            return self.mango_enricher.political_score
        else:
            return None

    @property
    def party_labour_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('labour_party', 0)
        else:
            return None

    @property
    def party_conservative_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('conservative_party', 0)
        else:
            return None

    @property
    def party_libdems_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('liberal_democrats', 0)
        else:
            return None

    @property
    def party_snp_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('scottish_national_party', 0)
        else:
            return None

    @property
    def party_dup_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('democratic_unionist_party', 0)
        else:
            return None

    @property
    def party_sinnfein_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('sinn_féin', 0)
        else:
            return None

    @property
    def party_plaidcymru_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('plaid_cymru', 0)
        else:
            return None

    @property
    def party_greens_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('green_party', 0)
        else:
            return None

    @property
    def party_sdu_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('social_democratic_unionists', 0)
        else:
            return None


    @property
    def party_alliance_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('alliance_party_of_northern_ireland', 0)
        else:
            return None

    @property
    def party_brexit_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('brexit_party', 0)
        else:
            return None

    @property
    def party_ukip_refs(self):
        if self.mango_enricher:
            return self.mango_enricher.political_party_refs.get('uk_independence_party', 0)
        else:
            return None

    @property
    def asset_uri(self):
        return get_value_or_none_from_asset(self.data, ['metadata', 'locators', 'assetUri'])

    @property
    def type(self):
        return "asset"

    @property
    def site(self):
        return re.match(r"^/(.+?)/", self.asset_uri).group(1)  # /sport/football returns site="sport"

    @property
    def has_content_warning(self):
        return bool(get_value_or_none_from_asset(self.data, ['metadata', 'options', 'hasContentWarning']))

    @property
    def tags(self):
        all_tags = get_value_or_none_from_asset(self.data, ["metadata", "tags", "about"])
        tags_list = [{"id": tag_dict.get("thingId"), "label": tag_dict.get("thingLabel")}
                     for tag_dict in all_tags or []]
        tags_list = remove_redundant_tags(tags_list, [self.category_level_1, self.category_level_2])
        if tags_list:  # avoid returning empty list, returns None if no tags were found
            return tags_list

    @property
    def tags_text(self):
        concat_tags = ' '.join([tag.get('label', '').replace(' ', '_') for tag in self.tags or []])
        if concat_tags:
            return concat_tags

    @property
    def category_level_1(self):
        if self.asset_uri:
            return get_category_from_asset_uri(self.asset_uri, 0)

    @property
    def category_level_2(self):
        if self.asset_uri:
            return get_category_from_asset_uri(self.asset_uri, 1)

    @property
    def category_level_3(self):
        if self.asset_uri:
            return get_category_from_asset_uri(self.asset_uri, 2)

    @property
    def last_updated(self):
        timestamp = get_value_or_none_from_asset(self.data, ['metadata', 'lastUpdated'])
        if timestamp:
            return self.get_date_from_timestamp(timestamp)  # "lastUpdated":1564754800836

    @property
    def last_published(self):
        timestamp = get_value_or_none_from_asset(self.data, ['metadata', 'lastPublished'])
        if timestamp:
            return self.get_date_from_timestamp(timestamp)

    @property
    def first_published(self):
        timestamp = get_value_or_none_from_asset(self.data, ['metadata', 'firstPublished'])
        if timestamp:
            return self.get_date_from_timestamp(timestamp)

    @property
    def news_update_or_last_published(self):
        # TODO this is temporary, in ARES v2 it will be replaced by lastPublished
        timestamp = get_value_or_none_from_asset(self.data, ['metadata', 'timestamp'])
        if timestamp:
            return self.get_date_from_timestamp(timestamp)
        else:
            return self.last_published

    @property
    def headline(self):
        headline = get_value_or_none_from_asset(self.data, ['promo', 'headlines', 'headline'])
        if headline:
            return headline

    @property
    def short_headline(self):
        short_headline = get_value_or_none_from_asset(self.data, ['promo', 'headlines', 'shortHeadline'])
        if short_headline:
            return short_headline

    @property
    def summary(self):
        summary = get_value_or_none_from_asset(self.data, ['promo', 'summary'])
        if summary:
            return summary

    @property
    def body(self):
        """
        Get the text from all the [content][blocks] nests within an Ares asset.
        Args:
            asset: dictionary to retrieve body text from
        Returns:
            string of all body text from the Ares asset
        """

        body = ' '.join(
            [block.get("text", '') for block in get_value_or_none_from_asset(self.data, ['content', 'blocks']) or []]
        )
        cleaned_body = clean_html_markup(body)
        if cleaned_body:
            return cleaned_body

    @property
    def combined_body_summary_headline(self):
        to_combine = [block for block in [self.headline, self.summary, self.body] if block is not None]
        combined = ' '.join(to_combine)
        if combined:
            return combined

    @property
    def ad_campaign(self):
        ad_campaign_keyword = get_value_or_none_from_asset(self.data, ['metadata', 'adCampaignKeyword'])
        return ad_campaign_keyword is not None

    @staticmethod
    def get_date_from_timestamp(timestamp):
        if (timestamp / 1e11) > 1:
            timestamp = timestamp / 1000.

        dt = datetime.fromtimestamp(timestamp, tz=pytz.utc).strftime('%Y-%m-%dT%H:%M:%S%z')
        # Elasticsearch requires a colon between hours and minutes for time difference (I think!)
        dt_string = make_datetime_string_compatible_with_es_ingestion(dt)
        return dt_string


class SportAresItem(EnrichedAresItem):

    def __init__(self, ares_dict, mango_enrichment=False):
        super().__init__(ares_dict, mango_enrichment)
        self.filter_config = load_config_file(os.path.join(PARENT_DIR, "config", "sport_filter_config.yaml"))

    @property
    def tags_competition_text(self):
        '''
        Returns the list of tags that are defined as sport competitions, as a single string of text, separated by spaces
        Returns: str

        '''
        all_tags = get_value_or_none_from_asset(self.data, ["metadata", "tags", "about"])
        competition_tags_list = [tag_dict.get("thingLabel")
                                 for tag_dict in all_tags or [] if "sport:Competition" in tag_dict.get("thingType", [])]
        if competition_tags_list:  # avoid returning empty list, returns None if no tags were found
            competition_tags_text = ' '.join([tag.replace(' ', '_') for tag in competition_tags_list or []])
            return competition_tags_text

    @property
    def is_periodical(self):
        if self.headline:
            return any(periodical_indicator.lower() in self.headline.lower()
                       for periodical_indicator in self.filter_config['PERIODICAL_INDICATORS'])
        else:
            return False

    @property
    def is_red_button_content(self):
        return 'DIGIONLY' in self.summary

    @property
    def is_rankings(self):
        """
        Checks if short headline equals one of the ranking titles
        Returns: Bool

        """
        if self.short_headline:
            return self.short_headline in self.filter_config['RANKINGS_INDICATORS']

    @property
    def is_squad(self):
        """
        Checks if short headline equals one of the squad titles
        Returns: Bool

        """
        if self.short_headline:
            return self.short_headline in self.filter_config['SQUAD_INDICATORS']

    @property
    def is_competition_draw(self):
        """
        Checks if asset id is in list of ids defined as competition draws
        Returns: Bool

        """
        asset_id = self.asset_uri.split('/')[-1]
        if asset_id:
            return asset_id in self.filter_config['COMPETITION_DRAW_IDS']

    @property
    def is_match_report(self):
        """
        Checks if article is a match report or preview by looking up to one page per article metadata
        Returns: Bool

        """
        return bool(get_value_or_none_from_asset(self.data, ['metadata', 'isOppm']))


class NewsAresItem(EnrichedAresItem):

    def __init__(self, ares_dict, mango_enrichment=False):
        super().__init__(ares_dict, mango_enrichment)
        self.filter_config = load_config_file(os.path.join(PARENT_DIR, "config", "news_filter_config.yaml"))

    @property
    def article_category_name(self):
        """
        Extract the article category name from the asset uri if it exists
        Examples:
            '/russian/news-12345678' returns 'news'
            '/sport/football/' returns None

        Returns:
            (str) value of the article category name
        """
        article_category_name = get_article_type(self.asset_uri)
        return article_category_name.lower() if article_category_name else article_category_name

    @property
    def is_breaking_news(self):
        return bool(get_value_or_none_from_asset(self.data, ['metadata', 'options', 'isBreakingNews']))

    @property
    def is_uk_news(self):
        if self.article_category_name:
            return True if 'uk' in self.article_category_name else False

    @property
    def nation(self):
        if self.article_category_name:
            for nation in self.filter_config['NATIONS']:
                if nation in self.article_category_name:
                    return nation

    @property
    def is_press_review(self):
        for press_review_indicator in self.filter_config['PRESS_REVIEW_INDICATORS']:
            if self.headline:
                if press_review_indicator.lower() in self.headline.lower():
                    return True

            if self.short_headline:
                if press_review_indicator.lower() in self.short_headline.lower():
                    return True

        return False

    @property
    def is_legal(self):
        for legal_indicator in self.filter_config['LEGAL_INDICATORS']:
            if legal_indicator.lower() in self.combined_body_summary_headline.lower():
                return True

        if self.tags_text:
            for legal_tag in self.filter_config['LEGAL_TAGS']:
                if legal_tag.lower() in self.tags_text.lower():
                    return True

        return False

    @property
    def is_politics(self):
        if self.article_category_name:
            return True if 'politics' in self.article_category_name else False

    @property
    def is_elections(self):
        if self.article_category_name:
            return True if 'election' in self.article_category_name else False


def get_article_type(asset_uri, regex=r'([A-Za-z0-9\-]{1,})[0-9]{8}'):
    """
    Extract the article category name from the asset uri if it exists
    Examples:
        '/russian/news-12345678' returns 'news'
        '/sport/football/' returns None

    Args:
        asset_uri (str): assetURI of the article e.g. /russian/news-12345678
        regex (str): regular expression string used to identify the article category from the asset_uri
    Returns:
        str: string matching result
    """
    article_id = asset_uri.split('/')[-1]
    results = re.findall(regex, article_id)
    if results:
        result = results[0]
        return result[:-1] if result.endswith('-') else result  # remove trailing dash


def get_value_or_none_from_asset(asset, keys):
    """
    Iteratively goes through the keys fields to get the specified value
    Example:
        {'promo': {'headlines' : {'headline': 'Test', 'shortHeadline': 'Short Test'}}}
        keys of ['promo', 'headlines', 'shortHeadline'] would return 'Short Test'
    Args:
        asset (dict): dict to retrieve value from
        keys (iterable): iterable of keys to use to get value
    Returns:
        value if exists else None
    """
    value = asset
    for key in keys:
        value = value.get(key, {})
    return value or None


def get_category_from_asset_uri(asset_uri, category_idx):
    """
    Get the category from the asset_uri.
    Example:
        category_idx of 1 for '/sport/football/12345678' would return 'football'
    Args:
        asset_uri (str): string for asset uri e.g. '/sport/football/12345678'
        category_idx (int): index for the specific category
    Returns:
        (str) category value if category_idx < number of categories
    """
    categories = asset_uri.split('/')[1: -1]
    if category_idx < len(categories):
        return categories[category_idx]


def get_fields_from_ares_asset(asset):
    """
    Converts an Ares asset (i.e. a document) to a filtered dict of
    Args:
        asset: ares document json

    Returns:
        dict of the filtered Ares document with camel cased fields
    """
    site = asset.get('metadata', None).get('locators', None).get('assetUri', None).split('/')[1]
    # superclass_properties = [p for p in dir(EnrichedAresItem) if isinstance(getattr(EnrichedAresItem, p), property)]

    if site == 'sport':
        article = SportAresItem(asset)

    elif site == 'news':
        article = NewsAresItem(asset, True)
    else:
        raise ValueError('Only Sport and News ingestion is currently supported.')

    return to_camelcase_dict(article)


def make_datetime_string_compatible_with_es_ingestion(dt_string):
    if len(dt_string) is 24:
        es_dt_string = dt_string[:22] + ":" + dt_string[22:]
        return es_dt_string
    else:
        return dt_string


def remove_redundant_tags(all_tags: dict, labels_to_remove: list):
    """
    Takes list of dictionaries and removes any whose 'label' field matches a string specified in the second input
    param.
    Args:
        all_tags (list): List of dicts, each with the key 'label'
        labels_to_remove (list): Remove all dicts with label containing these strings

    Returns:
        list

    """
    filtered_tags = []
    labels_to_remove = [label.replace(' ', '-').lower() for label in labels_to_remove if label is not None]
    for tag in all_tags:
        if tag['label'].replace(' ', '-').lower() in labels_to_remove:
            continue
        filtered_tags.append(tag)

    return filtered_tags


def class_to_dict(instantiated_class):
    """Convert properties to a dictionary, skip properties with None values"""
    data_dict = {}
    property_list = get_class_properties(type(instantiated_class))
    for key in property_list:
        val = getattr(instantiated_class, key)
        val = remove_empty_elements(val)
        if val is not None:
            data_dict[key] = val

    return data_dict


def get_class_properties(uninstantiated_class):
    """
    Returns a list of property names associated with an input class
    Args:
        uninstantiated_class (class): Class (not an object) to identify the property names of

    Returns: list<str>

    """
    property_list = []
    for prop in dir(uninstantiated_class):
        try:
            if isinstance(getattr(uninstantiated_class, prop), property):
                property_list.append(prop)

        except AttributeError:
            pass
    return property_list


def to_camelcase_dict(instantiated_class):
    """Rename properties from snake_case to camelCase as required by the ES mapping."""
    return convert_dict_underscore_keys_to_camelcase(class_to_dict(instantiated_class))
