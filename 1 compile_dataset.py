# -*- coding: utf-8 -*-
"""
Created on Sun April 12 21:07:10 2023

@author: Lim Jie Jones
"""

import pandas as pd
import re
import os


import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import emoji
import string

# For each datasets
    # Format each dataset
        # Keep only 3 var-content, label,tweet/article
        # add var for some that dont have tweeet/article 
def clean_data(dataset, varLabel, varContent):
    # Remove rows with empty values in varLabel column
    dataset = dataset.dropna(subset=[varLabel])
    dataset = dataset.dropna(subset=[varContent])

    return dataset
    
    # Clean out weirdly unicode error thingamabob
    # Dont need cus can encode when reading

def replace_source_values(df, column_name):
    social_media = [
        'Social media','socialmedia','social_media','social-media','Youtube', 'Twitter', 'Facebook', 'WhatsApp', 'Weibo', 'Instagram', 'TikTok',
        'Soundcloud', '4Chan', 'Kakao Talk', 'Line', 'Niconico', 'Telegram', 'Reddit',
        'Vimeo', 'LinkedIn', 'Douyin', 'Xigua Video', 'Biohack.info', 'WION', 'ET Now',
        'Viber'
    ]

    df = df.dropna(subset=[column_name])  # Drop rows with missing or NaN values in the specified column
    df[column_name] = df[column_name].apply(lambda x: 'social media' if any(keyword.lower() in x.lower() for keyword in social_media) else 'articles')
    # df[column_name] = df[column_name].apply(lambda x: 'social media' if any(re.search(r'\b{}\b'.format(re.escape(keyword)), x, flags=re.IGNORECASE) for keyword in social_media) else 'articles')
    # df[column_name] = df[column_name].apply(lambda x: 'articles' if x.startswith('http') or '.' in x else x)
    return df

def dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
    dataset=datasetOG[[varTitle,varContent,varSource,varLabel]]
    
    
    if not MonoSourceCheck:
        dataset = replace_source_values(dataset,varSource)
    
    # Clean data: Remove rows with empty values in varLabel column
    dataset = clean_data(dataset, varLabel, varContent)
    # Rename variables in the dataset
    dataset = dataset.rename(columns={varTitle: 'title', varContent: 'content', varSource: 'source', varLabel: 'label'})
    
    return dataset

def clean_dataset2(dataset):
    # return dataset
    column_name='label'
    dataset[column_name] = dataset[column_name].str.lower()

    # List of words to be replaced with 1
    words_to_replace = ['true', 'correct attribution', 'mostly true', 'news', 'collections','real']

    # Create a regex pattern to match the words
    pattern = '|'.join(words_to_replace)

    # Remove rows with '(Org. doesn't apply rating)' as their label
    dataset = dataset[~dataset[column_name].str.contains(r'\(Org\. doesn\'t apply rating\)', na=False)]

    # Replace the matching words with 1 using regex
    dataset[column_name] = dataset[column_name].apply(lambda x: 1 if re.search(pattern, str(x)) else 0)

    # Convert the remaining labels to 0
    # dataset[column_name] = dataset[column_name].replace(to_replace='.*', value=0, regex=True)
    dataset[column_name] = dataset[column_name].apply(lambda x: 0 if x != 1 else x)

    # Convert the column to numeric type
    # dataset[column_name] = pd.to_numeric(dataset[column_name])

    return dataset

# Could probably automate this. But I wont. Cus that will take longer time
# # Dataset 1
# encoding = 'utf-8'  # Specify the desired encoding
# folder_path = 'Cap2 Datasets/Cap2 Datasets/'
# file_name = '(1 3) fake_new_dataset.xlsx'
# # encodings = ['utf-8', 'latin-1', 'utf-16', 'cp1252', 'iso-8859-1']

# filename=os.path.join(os.getcwd(),folder_path,file_name)

# ############################################################################
# df = pd.read_excel(filename)

# df['source']='articles'
# data1=dataset_sort(df,df.columns[1],df.columns[2],df.columns[5],df.columns[4],True)

# ##########################################################################################
# # Dataset 2
# folder_path = 'Cap2 Datasets/Cap2 Datasets/'
# file_name = '(4 5) FakeCovid_July2020.csv'
# os.getcwd()
# filename=os.path.join(os.getcwd(),folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['source']='articles'
# df=df[df['lang'] == 'en']
# data2=dataset_sort(df,'title','content_text',
#                     'source','class',True)
# pd.unique(data2.label)
# # Define the list of keywords
# data2_clean=clean_dataset2(data2)
# pd.unique(data2_clean.label)

# ##############################################################################################
# # Dataset 3 (7) no need clean
# folder_path = './Cap2 Datasets/Cap2 Datasets/'
# file_name = '(7) COVID Fake News Data.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['source']='articles'
# df['content']=''
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data3=dataset_sort(df,df[['content']].columns[0],df[['headlines']].columns[0],
#                     df[['source']].columns[0],df[['outcome']].columns[0],True)
# ###############################################################################################
# # Dataset 4 (10)
# folder_path = './Cap2 Datasets/Cap2 Datasets/'
# file_name = '(10) jns-covid_misinfo_2021-03-06_Final_Clean.xlsx'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_excel(filename)
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# pd.unique(df.Distrib_Channel)
# data4=dataset_sort(df,'Title','Narrative_Description',
#                     'Distrib_Channel','Misinfo_Type',False)

# data4['label']=0
# ############################################################################################

# # Dataset 5 (2)
# folder_path = './Cap2 Datasets/Cap2 Datasets/'
# file_name = 'Constraint_English_Train - Sheet1.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['source']='social media'
# df['title']=''
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data5_1=dataset_sort(df,'title','tweet',
#                     'source','label',True)
# data5_1_clean=clean_dataset2(data5_1)
# pd.unique(data5_1_clean.label)

# file_name = 'Constraint_English_Val - Sheet1.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['source']='social media'
# df['title']=''
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data5_2=dataset_sort(df,'title','tweet',
#                     'source','label',True)
# data5_2_clean=clean_dataset2(data5_2)
# pd.unique(data5_2_clean.label)

# ############################################################################################

# # Dataset 6 (6)
# folder_path = './Cap2 Datasets/Cap2 Datasets/(6) CoAID-master/05-01-2020/'
# file_name = 'NewsFakeCOVID-19.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['label']='0'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data6_1=dataset_sort(df,'title','content',
#                     'fact_check_url','label',False)
# file_name = 'NewsRealCOVID-19.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['label']='1'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data6_2=dataset_sort(df,'title','content',
#                     'fact_check_url','label',False)
# ################################
# folder_path = './Cap2 Datasets/Cap2 Datasets/(6) CoAID-master/07-01-2020/'
# file_name = 'NewsFakeCOVID-19.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['label']='0'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data6_3=dataset_sort(df,'title','content',
#                     'fact_check_url','label',False)
# file_name = 'NewsRealCOVID-19.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['label']='1'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data6_4=dataset_sort(df,'title','content',
#                     'fact_check_url','label',False)
# ################################
# folder_path = './Cap2 Datasets/Cap2 Datasets/(6) CoAID-master/09-01-2020/'
# file_name = 'NewsFakeCOVID-19.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['label']='0'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data6_5=dataset_sort(df,'title','content',
#                     'fact_check_url','label',False)
# file_name = 'NewsRealCOVID-19.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# df['label']='1'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data6_6=dataset_sort(df,'title','content',
#                     'fact_check_url','label',False)

# # ############################################################################################
# # # Dataset 7 (14 15)
# # folder_path = './Cap2 Datasets/Cap2 Datasets/(14 15) formal-sources/'
# # file_name = 'public_advices-from-credible_sources.csv'
# # filename=os.path.join(folder_path,file_name)
# # df = pd.read_csv(filename,encoding=encoding)
# # df['source']='articles'
# # df['title']=''
# # df['label']='1'
# # # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# # data7_1=dataset_sort(df,'title','Text',
# #                     'source','label',True)
# # file_name = 'WHO-public-advice.csv'
# # filename=os.path.join(folder_path,file_name)
# # df = pd.read_csv(filename,encoding=encoding)
# # df['source']='articles'
# # df['title']=''
# # df['label']='1'
# # # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# # data7_2=dataset_sort(df,'title','Text',
# #                     'source','label',True)

# ############################################################################################
# # Dataset 8 (ReCovery)
# folder_path = './Cap2 Datasets/Cap2 Datasets/'
# file_name = 'ReCOVery_master_dataset_recovery-news-data.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# # df['source']='articles'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data8=dataset_sort(df,'title','body_text',
#                     'url','reliability',False)

# ############################################################################################
# # Dataset 9 (FaCov)
# folder_path = './Cap2 Datasets/Cap2 Datasets/'
# file_name = 'FaCov_dataset.csv'
# filename=os.path.join(folder_path,file_name)
# df = pd.read_csv(filename,encoding=encoding)
# # df['source']='articles'
# # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# data9=dataset_sort(df,'title','content',
#                     'title-href','2class_labels',False)
# data9['label']=data9['label'].astype(str)
# data9_clean=clean_dataset2(data9)

# # ############################################################################################
# # # The Dataset
# # folder_path = './Cap2 Datasets/Cap2 Datasets/'
# # file_name = 'Cord19 metadata.csv'
# # filename=os.path.join(folder_path,file_name)
# # df = pd.read_csv(filename,encoding=encoding)
# # df['label']=1
# # df['source']='articles'
# # # #  dataset_sort(datasetOG,varTitle,varContent,varSource,varLabel,MonoSourceCheck):
# # dataA=dataset_sort(df,'title','abstract',
# #                     'source','label',True)
# # # data9['label']=data9['label'].astype(str)
# # # data9_clean=clean_dataset2(data9)
# # #######################################################################################

# # dataA.to_csv('Big_Research_data.csv', index=False,mode='w+' )


# # Combine everything
# combined_df = pd.concat([data1, data2_clean,data3,data4,
#                          data5_1_clean,data5_2_clean,
#                          data6_1,data6_2,data6_3,data6_4,data6_5,data6_6,
#                          data8,data9_clean], ignore_index=True)
# combined_df['label']=combined_df['label'].astype(str)
# for i in range(len(combined_df['label'])):
#     if combined_df['label'][i] == '1':  # Check if the value is '1' as a string
#         combined_df['label'][i] = 1  # Convert '1' to the integer value 1
#     if combined_df['label'][i] == '0':  # Check if the value is '1' as a string
#         combined_df['label'][i] = 0  # Convert '1' to the integer value 1
# combined_df=combined_df.drop_duplicates()

# # Saving the combined dataframe as a CSV file
# combined_df.to_csv('combined_data.csv', index=False,mode='w+' )

# # check ratio
# combined_df['label'].value_counts(normalize=False)

# check obs no.

# ==================================================================================================================        
# ==================================================================================================================        
# ==================================================================================================================        



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
cdf = pd.read_csv('combined_data.csv')
content_column = cdf['content']

# Preprocess the text data
cleaned_tokens = [preprocess_text(text) for text in content_column]

# Create dictionary and corpus
id2word = corpora.Dictionary(cleaned_tokens)
id2word.filter_extremes(no_below=30, no_above=30)
corpus = [id2word.doc2bow(text) for text in cleaned_tokens]

# Build the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            passes=50,
                                            iterations=50,
                                            num_topics=5,
                                            random_state=1)
coherence_model_lda=CoherenceModel(model=ldamodel,texts=cleaned_tokens,dictionary=id2word,coherence='u_mass')
# coherence_model_lda=CoherenceModel(model=ldamodel,texts=cleaned_tokens,dictionary=id2word,coherence='u_mass')
coherence_lda=coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
print('\nCoherence Score: ', coherence_model_lda.get_coherence())

# Assign topics to each document
topics = [sorted(ldamodel[doc], key=lambda x: x[1], reverse=True)[0][0] for doc in corpus]

ldamodel.print_topics(num_words=15)
# Create a new column 'Topic' in the dataframe
cdf['Topic'] = topics
vacdf=cdf[cdf['Topic']==3]
# vacdf = cdf[cdf['Topic'].isin([3, 5])]
vacdf['label'].value_counts(normalize=False)
vacdf=vacdf.drop_duplicates()
vacdf.to_csv('combined_sortedbytopic.csv', index=False,mode='w+' )
