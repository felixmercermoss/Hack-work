import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# use the compound score

sid = SentimentIntensityAnalyzer()

data_path = '/Users/harlaa04/workspace/Hack-work/HACK-2020-Q1/data/sentiment/IMDB Dataset.csv' # annoyingly, no neutral examples...


data = pd.read_csv(data_path)
print(data.head())
print(data.groupby('sentiment').count())

sentiment_mapper = {'positive': 1, 'negative': 0}

data['encoded_sentiment'] = data['sentiment'].apply(lambda x: sentiment_mapper[x])



vader_mapper_0_1 = lambda x: 0 if x < 0. else 1

data['vader_scores'] = data['review'].apply(lambda x: sid.polarity_scores(x).get('compound', 0.))
data['vader_encoded'] = data['vader_scores'].apply(vader_mapper_0_1)

data['matches'] = data['encoded_sentiment'] == data['vader_encoded']

print(data)

matches = data.groupby('matches')['sentiment'].count()

print(matches)

score = matches.loc['true'] / len(data) * 100 # this line aint working 

print(f'Accuracy of vader on imdb :{score}')

print(data.describe())
