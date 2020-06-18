import logging
import re
import requests

import gender_guesser.detector as gender
from SPARQLWrapper import SPARQLWrapper, JSON
from statistics import mean


PEOPLE_ENRICHMENT_PREDICATES = ['dbo:party', 'foaf:gender']
PARTY_ENRICHMENT_PREDICATES = ['dbp:position']
PERSON_TYPE = 'http://dbpedia.org/ontology/Person'
POLITICAL_PARTY_TYPE = 'http://dbpedia.org/ontology/PoliticalParty'
POLITICIAN_TYPE = 'http://dbpedia.org/ontology/OfficeHolder'
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
gender_predictor = gender.Detector(case_sensitive=False)

POLITICAL_MAPPER = {'left wing': -2,
                    'centre left': -1,
                    'center left': -1,
                    'centrism': 0,
                    'centre right': 1,
                    'center right': 1,
                    'right wing': 2}
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()


def fuzzy_match_political_position(pos):
    if pos:
        regex = re.compile('[^a-zA-Z]')
        pos = regex.sub(' ', pos).lower()
        for p in POLITICAL_MAPPER:
            if p in pos:
                return POLITICAL_MAPPER[p]
    else:
        return None


def request_gender_from_gender_detector(first_name):
    predicted_gender = gender_predictor.get_gender(first_name, country='great_britain')
    if predicted_gender == 'mostly_female':
        predicted_gender = 'female'

    if predicted_gender == 'mostly_male':
        predicted_gender = 'male'

    return {'value': predicted_gender}


class enrichedMangoTags():
    def __init__(self, uri):
        self.tags = self.fetch_mango_enrichment(uri)
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat(JSON)
        self.people_tags = self.filter_tags_by_type(self.tags, PERSON_TYPE)
        self._enrich_people_tags()
        self.political_parties_referenced = self.get_political_parties_referenced()
        self._enrich_political_parties()
        self.political_party_refs = self.num_political_party_references()

    @staticmethod
    def filter_tags_by_type(tags, entity_type):
        return [tag for tag in tags if entity_type in tag.get('types')]

    def fetch_dbpedia_metadata(self, subject_uri, predicates):
        metadata = {}
        for pred in predicates:

            query = construct_ssparql_query(subject_uri, [pred])
            self.sparql.setQuery(query)
            try:
                results = self.sparql.query().convert()
                data = results.get('results', {}).get('bindings', [])
                if data:
                    metadata.update(data[0])
            except Exception as e:
                logger.error(e)

        return metadata

    def fetch_mango_enrichment(self, uri):
        url = f'http://api.mango-en.virt.ch.bbc.co.uk/topics?uri=https://www.bbc.co.uk{uri}'
        data = requests.get(url)
        return data.json()['results']

    def _enrich_people_tags(self):
        for tag in self.people_tags:
            metadata = self.fetch_dbpedia_metadata(tag.get('uri'), PEOPLE_ENRICHMENT_PREDICATES)
            tag.update(metadata)
            first_name = tag.get('label').split(' ')[0]
            if 'gender' not in tag:

                tag['gender'] = request_gender_from_gender_detector(first_name)

            if tag.get('gender', {}).get('value', '') not in set(['male', 'female']):
                tag['gender'] = request_gender_from_genderize(first_name)

    def _enrich_political_parties(self):
        for party in self.political_parties_referenced:
            metadata = self.fetch_dbpedia_metadata(party.get('uri'), PARTY_ENRICHMENT_PREDICATES)
            party.update(metadata)

            if 'position' not in party:
                metadata = self.fetch_dbpedia_metadata(party.get('uri'), ['dbp:successor'])
                successor_uri  = metadata.get('successor', {}).get('value', {})
                if successor_uri:
                    metadata = self.fetch_dbpedia_metadata(successor_uri, PARTY_ENRICHMENT_PREDICATES)
                    party.update(metadata)

            if 'position' not in party:
                metadata = self.fetch_dbpedia_metadata(party.get('uri'), ['dbo:wikiPageRedirects'])
                redirect_uri = metadata.get('wikiPageRedirects', {}).get('value', {})
                if redirect_uri:
                    metadata = self.fetch_dbpedia_metadata(redirect_uri, PARTY_ENRICHMENT_PREDICATES)
                    party.update(metadata)

    def get_list_of_political_positions(self):
        politicians = []
        for p in self.politicians_mentioned:
            party = [p.get('party', {}).get('value', None)] * p.get('count', 1)
            politicians = politicians + party

        parties = []
        for p in self.political_parties_mentioned:
            party = [p.get('uri', None)] * p.get('count', 1)
            parties = parties + party

        all_parties = politicians + parties

        positions = []
        for party in all_parties:
            position =  [p.get('position', {}).get('value') for p in self.political_parties_referenced if p.get('uri', "") == party]
            if position:
                positions.append(position[0])

        return positions

    @property
    def political_score(self):
        positions = self.get_list_of_political_positions()
        scores = []
        for pos in positions:
            scores.append(fuzzy_match_political_position(pos))
        score_no_none = [s for s in scores if s]
        if len(score_no_none) > 0:
            return mean(score_no_none)
        else:
            return None

    @property
    def num_political_references(self):
        return len(self.get_list_of_political_positions())



    def num_political_party_references(self):
        party_counts = {}
        for ref in self.political_parties_referenced:
            if ref:
                party_referenced = [p for p in POLITICAL_PARTIES if p in ref.get('uri', '').lower()]
                if party_referenced and party_referenced[0] not in party_counts:
                    party_counts.setdefault(party_referenced[0], 0)
                    party_counts[party_referenced[0]] += 1

        return party_counts

    @property
    def female_mentions_proportion(self):
        female_tags = [p for p in self.people_mentioned if p and p.get('gender','').get('value', '') == 'female']
        num_female_mentions = sum([int(p.get('count', 0)) for p in female_tags])

        male_tags = [p for p in self.people_mentioned if p and p.get('gender','').get('value', '') == 'male']
        num_male_mentions = sum([int(p.get('count', 0)) for p in male_tags])

        if (num_female_mentions + num_male_mentions) > 0 :
            return num_female_mentions / (num_female_mentions + num_male_mentions)
        else:
            return None

    def get_political_parties_referenced(self):
        political_party_references = []
        for politician in self.politicians_mentioned:
            political_party_references.append(politician.get('party', {}).get('value', None))

        for political_party in self.political_parties_mentioned:
            political_party_references.append(political_party.get('uri', None))

        political_party_references = filter(None, political_party_references)
        political_parties_referenced = [{'uri': party} for party in set(political_party_references)]
        return political_parties_referenced

    @property
    def people_mentioned(self):
        return self.people_tags

    @property
    def political_parties_mentioned(self):
        return self.filter_tags_by_type(self.tags, POLITICAL_PARTY_TYPE)

    @property
    def politicians_mentioned(self):
        return self.filter_tags_by_type(self.people_mentioned, POLITICIAN_TYPE)


def construct_ssparql_query(subject, predicates):
    vars = ['?' + v.split(':')[1] for v in predicates]
    where_clauses = []
    for pred, vari in zip(predicates, vars):
        where_clauses.append(f'{pred} {vari} ')

    query = f"""
    SELECT {' '.join(vars)}
        WHERE {{
            <{subject}> {';'.join(where_clauses)} .
                }}
    """
    return query



def test_fetch_mango_enrichment():
    uri = "/news/uk-england-london-51628623"
    #uri = '/news/uk-politics-51665162'
    mango_tags = enrichedMangoTags(uri)
    mango_tags.political_score
    pass

def test_request_gender_from_genderize():
    predicted_male = request_gender_from_genderize('felix')
    assert predicted_male.get('value') == 'male'

    predicted_female = request_gender_from_genderize('siobahn')
    assert predicted_female.get('value') == 'female'


def request_gender_from_genderize(first_name):
    url = f'https://api.genderize.io?name={first_name}&country_id=GB'
    response = requests.get(url)
    if response.status_code is 200:
        data = response.json()
        gender =  data.get('gender')
        return {'value': gender, 'probability': data.get('gender')}
    else:
        return {'value': None}


