"""
This file contains functions which will identify a subjects tone towards an object.


If we consider author tone towards the subject, we should look at the following criteria:
 - sentiment towards the subject
 - emotional response to the subject: anger, disbelief etc.
 - presence of sarcasm
 - ???
"""
import spacy

def extract_subject_object_adj(text):
    pass


def detect_tone(triplet):
    pass


nlp = spacy.load('en_core_web_sm')

doc = nlp("Autonomous cars shift insurance liability toward manufacturers")

for chunk in doc.noun_chunks:
    print(f'chunk.text: {chunk.text}')
    print(f'chunk.root.text: {chunk.root.text}')
    print(f'chunk.root.dep_: {chunk.root.dep_}')
    print(f'chunk.root.head.text: {chunk.root.head.text}')
    print('============')



"""
# algo design - article tonality
# identify n-degrees of tonality: sentiment, emotions dist,
classfier for each of the tasks.

For sentiment, use spacy built in and generate a dist of the sentiment across the article

For others, take sentence-bert and convert each sentence to single vector
Build a classifier associated to the task (SVM?).
Predict probs across each of the classes for each sentence, and do this across the full article

"""
