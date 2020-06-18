def convert_dict_underscore_keys_to_camelcase(dict_obj):
    """
    Recursive function to convert dicts to camelcase, including nested dicts.
    Args:
        dict_obj: {'under_score': '', 'tags': [{'thing_id': '', 'thing_label': ''}]}

    Returns:
        {'assetUri: 'tags': [{'thingId': '', 'thingLabel': ''}]}
    """
    d_new = {}
    for key_underscore, value in dict_obj.items():
        if isinstance(value, list):
            value = [convert_dict_underscore_keys_to_camelcase(element) for element in value]
        elif isinstance(value, dict):
            value = convert_dict_underscore_keys_to_camelcase(value)
        d_new[underscore_to_camelcase(key_underscore)] = value
    return d_new


def camelcase():
    """
    Yields:
        str functions to convert a str to camelcase
    """
    yield str.lower
    while True:
        yield str.capitalize


def underscore_to_camelcase(value):
    """
    Converts underscore separated tokens to camelcase, splitting tokens by `_`
    Args:
        value: string to convert to camelcase

    Returns:
        string representing camelcased input
    """
    camelcase_func = camelcase()
    return "".join(next(camelcase_func)(x) if x else '_' for x in value.split("_"))


def remove_empty_elements(obj):
    """
    Recursively removes empty/None elements from dicts and lists
    Args:
        obj: object to remove empty/None elements from

    Returns:
        object with all empty/None elements removed.
    """
    if isinstance(obj, list):
        obj = [remove_empty_elements(element) for element in obj]
    elif isinstance(obj, dict):
        no_empty_elements_obj = {}
        for key, value in obj.items():
            empty_removed_values = remove_empty_elements(value)
            if empty_removed_values:
                no_empty_elements_obj[key] = empty_removed_values
        obj = no_empty_elements_obj
    if obj is not None:
        return obj
