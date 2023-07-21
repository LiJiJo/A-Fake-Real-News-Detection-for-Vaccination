# V3 
# -*- coding: utf-8 -*-

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np


import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import emoji
import string

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import load_model


import joblib

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
# cdf = cdf.drop(['title', 'Topic'], axis=1)
# content_column = cdf['content']
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

# Train Word2Vec model
model = Word2Vec(sentences=cleaned_sentences, vector_size=500, window=5, min_count=1, workers=4)
Word2Vec_vector=model.wv.vectors
# Extract Word2Vec embeddings for each text
X = [np.mean([model.wv[token] for token in token_list], axis=0) for token_list in cleaned_sentences]
# Reshape X_array to add a new axis
X_df = pd.DataFrame(X)
# Concatenate X_array and encoded_source_array along axis 1
X_concatenated = np.hstack((np.array(X), encoded_source['articles'].values.reshape(-1, 1), encoded_source['social media'].values.reshape(-1, 1)))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_concatenated, cdf['label'], test_size=0.2, random_state=69
)
X_trainV, X_testV, y_trainV, y_testV = train_test_split(
    content_column, cdf['label'], test_size=0.2, random_state=69
)

np.save('./Test_data of topic sorted dataset/X_testV.npy', X_testV)
np.save('./Test_data of topic sorted dataset/X_test.npy', X_test)
np.save('./Test_data of topic sorted dataset//y_test.npy', y_test)
#################################################


# # Generate a synthetic dataset for demonstration
# X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create a neural network model
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=502, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create multiple neural network models
models = []
for _ in range(3):
    model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)
    models.append(model)

# Create the ensemble model using VotingClassifier
ensemble = VotingClassifier(estimators=[('model1', models[0]), ('model2', models[1]), ('model3', models[2])], voting='hard')
y_train_notseries = y_train.values
# Fit the ensemble model on the training data
ensemble.fit(X_train, y_train_notseries)

# Make predictions on the testing data
y_pred = ensemble.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("3NN Confusion Matrix:")
print(confusion)


#####################################################################################
# Define the RNN model
model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save('rnn_model.h5')
model = load_model('rnn_model.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

rnn_predictions = model.predict(X_test)
rnn_predictions = (rnn_predictions > 0.5).astype(int) 
confusion = confusion_matrix(y_test, rnn_predictions)
print("RNN Confusion Matrix:")
print(confusion)


########################################################################################

nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
                         random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
print("Neural Network Accuracy:", nn_accuracy)
confusion = confusion_matrix(y_test, nn_predictions)
print("NN Confusion Matrix:")
print(confusion)

# Logistic Regression
lr_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)
confusion = confusion_matrix(y_test, lr_predictions)
print("LR2 Confusion Matrix:")
print(confusion)

# Random Forest
# rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
#                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
#                                   bootstrap=True, oob_score=False, n_jobs=None, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=1, bootstrap=True, oob_score=False, n_jobs=None, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
confusion = confusion_matrix(y_test, rf_predictions)
print("RF Confusion Matrix:")
print(confusion)

# XGBoost
xgb_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, objective='binary:logistic',
                              booster='gbtree', tree_method='auto', n_jobs=1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("XGBoost Accuracy:", xgb_accuracy)
confusion = confusion_matrix(y_test, xgb_predictions)
print("XGB Confusion Matrix:")
print(confusion)

# Decision Tree
dt_model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                                  random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_report = classification_report(y_test, dt_predictions)
dt_confusion = confusion_matrix(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Classification Report:")
print(dt_report)
print("Decision Tree Confusion Matrix:")
print(dt_confusion)

######################################################

# Save each individual model within the ensemble
# joblib.dump(nn_model, 'nn_model.pkl')
# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(xgb_model, 'xgb_model.pkl')
# joblib.dump(lr_model, 'lr_model.pkl')

# Load each individual model
nn_model = joblib.load('nn_model.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
lr_model = joblib.load('lr_model.pkl')

# Create the ensemble model using VotingClassifier
ensemble = VotingClassifier(estimators=[
    ('neural_network', nn_model),
    ('random_forest', rf_model),
    ('xgboost', xgb_model),
    ('logistic_reg',lr_model)
], voting='hard')

# Fit the ensemble model on the training data
ensemble.fit(X_train, y_train)

# Make predictions on the testing data using the ensemble model
ensemble_predictions = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print("Ensemble Accuracy:", ensemble_accuracy)
confusion = confusion_matrix(y_test, ensemble_predictions)
print("Ensemble2 Confusion Matrix:")
print(confusion)

############################################################################
#####################################################################

