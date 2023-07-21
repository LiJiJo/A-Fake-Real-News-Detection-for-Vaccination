# V3 
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np


import re
# import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import emoji
import string


stop_words=stopwords.words('english')
def remove_noise(content_tokens, stop_words):
    cleaned_tokens=[]
    for token in content_tokens:
        token = re.sub('http([!-~]+)?','',token)
        token = re.sub('//t.co/[A-Za-z0-9]+','',token)
        token = re.sub('(@[A-Za-z0-9_]+)','',token)
        token = re.sub('[0-9]','',token)
        token = re.sub('[^ -~]','',token)
        token = emoji.replace_emoji(token, replace='')
        token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
        token = re.sub('[^\x00-\x7f]','', token) 
        token = re.sub(r"\s\s+" , " ", token)
        if (len(token)>3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def lemmatize_sentence(token):
    # initiate wordnetlemmatizer()
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence=[]
    
    # each of the words in the doc will be assigned to a grammatical category
    # part of speech tagging
    # NN for noun, VB for verb, adjective, preposition, etc
    # lemmatizer can determine role of word 
        # and then correctly identify the most suitable root form of the word
    # return the root form of the word
    for word, tag in pos_tag(token):
        if tag.startswith('NN'):
            pos='n'
        elif tag.startswith('VB'):
            pos='v'
        else:
            pos='a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
    return lemmatized_sentence
# Preprocessing function to tokenize, lemmatize, and remove noise
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemma_tokens = lemmatize_sentence(tokens)
    cleaned_tokens = remove_noise(lemma_tokens, stop_words)
    return cleaned_tokens

# Load the new dataset and access the 'content' column
# cdf=combined_df
cdf = pd.read_csv('combined_sortedbytopic.csv')  
content_column = cdf['title'].fillna('') + cdf['content']
encoded_source=pd.get_dummies(cdf['source'])

################################################################
# Preprocess the text data
cleaned_sentences = [preprocess_text(text) for text in content_column]

# Create a set of unique words from the cleaned sentences
word_set = set([word for sentence in cleaned_sentences for word in sentence])

# Create the lexicon with word-to-index mapping
lexicon = {word: index for index, word in enumerate(word_set)}

# Clean the sentences by removing words not present in the lexicon
cleaned_sentences = [[word for word in sentence if word in lexicon] for sentence in cleaned_sentences]
cleaned_sentences = [' '.join(tokens) for tokens in cleaned_sentences]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_sentences, cdf['label'], test_size=0.2, random_state=69
)

np.save('./Test_train for CTBERT/X_train_string.npy', X_train)
np.save('./Test_train for CTBERT/y_train.npy', y_train)
np.save('./Test_train for CTBERT/X_test_string.npy', X_test)
np.save('./Test_train for CTBERT/y_test.npy', y_test)