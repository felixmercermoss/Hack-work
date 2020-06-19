import re

import ruamel.yaml as yaml

from bs4 import BeautifulSoup



def get_html_tag_regex(tag):
    """
    Creates a regex to find a tag block e.g. `<p>here is some text</p>` could be found using tag='p'
    Args:
        tag: html tag string to generate regex for
    Returns:
        regex to find tag and all text between
    """
    html_regex = r'<{0}.*?>(.*?)<\/{0}>'
    return html_regex.format(tag)


def get_multiple_html_tag_finding_regex(tags):
    """
    combines tags into regex separated by | of the form <tags[i].*?>(.*?)</tags[i]>
    Args:
        tags: an iterable of strings to build into a regex of the form
        '<tags[0].*?>(.*?)</tags[0]>|<tags[1].*?>(.*?)</tags[1]>...'
    Returns:
        pipe char separated regex of the form '<tags[0].*?>(.*?)</tags[0]>|<tags[1].*?>(.*?)</tags[1]>...'
    """
    return '|'.join([get_html_tag_regex(tag) for tag in tags])


def remove_specified_asset_tags(raw_text, asset_tags=[]):
    """
    Removes the tag blocks (e.g. <p>...</p>) from the raw_text input for each of the asset_tags
    Args:
        raw_text: text/HTML to remove the asset tags from
        asset_tags: asset tags to remove
    Returns:
        raw_text with specified asset tags removed.
    """
    if asset_tags:
        removal_regex = get_multiple_html_tag_finding_regex(asset_tags)
        return re.sub(removal_regex, '  ', raw_text)
    else:
        return raw_text


def beautify_html(html):
    """
    Convert HTML to raw text using BeautifulSoup
    Args:
        html: HTML to be beautified
    Returns:
        Beautified HTML
    """
    soup = BeautifulSoup(html, 'html.parser')
    return soup.text


def clean_html_markup(body, asset_tags_to_remove=[]):
    """
    Clean the markup in the body text by removing specified asset tags and also beautifying the HTML using beautifulsoup.
    Args:
        body: markup representation of the body of the article
    Returns:
        string representing cleaned body content
    """
    html_minus_tags = remove_specified_asset_tags(body, asset_tags=asset_tags_to_remove)
    beautiful_text = beautify_html(html_minus_tags)
    return ' '.join(beautiful_text.split())


def load_config_file(yaml_path):
    """
    Load the config yaml file as dictionary
    Args:
        config_fpath: filepath of yaml config file

    Returns:
        dictionary of loaded config
    """
    with open(yaml_path) as yaml_file:
        yaml_loaded = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_loaded


def read_json(path_to_json):
    with open(path_to_json, 'rb') as f:
        data = json.load(f)
    return data