# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:19:47 2019

@author: elena.zotova
"""

import gensim, logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np

#Getting text corpus from our tweets. We cleaned it before. 
#I use lemmatized text, because all the words in the word2vec model are in their first form.
tweets = pd.read_csv('tweets_clean.csv', sep='\t')
tweets = tweets.fillna('')
tweets_text = tweets.text_lemmatized.values
tweets_text = list(tweets.text_lemmatized.values)

#Converting strings to tokens 
text_data = []
for line in tweets_text: 
    tokens = line.split()
    text_data.append(tokens)
    
#Loading pretrained word2vec model. I use Taiga Fasttext model from https://rusvectores.org/ru/
model = gensim.models.fasttext.FastTextKeyedVectors.load("w2vec_Araneum/araneum_none_fasttextcbow_300_5_2018.model")
w2v_vectors = model.wv.vectors 
w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab} 
num_features = model.vector_size
vocab = model.vocab.keys()

#How many words are there in the model? 
print(len(model.vocab))
#How similar are the words?
print (model.similarity('новость', 'новый'))
print (model.most_similar(positive=['камень'], negative=[], topn=2))

#For clustering tweets, we need to get a vector per tweet. This vector will be a mean of all vectors of the words of each tweet. 
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    not_in_model = []
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            v = model[word]
            #print(v)
            if np.isnan(v).any():
                print(word, v)
            featureVec = featureVec + model[word]
        else:
           not_in_model.append(word)
    #Here we can see if some of the words are not in the model, if so, we cannot use them for the clustering       
    print(not_in_model)
    # Dividing the result by number of words to get average
    if nwords != 0:
        featureVec = featureVec/nwords
    return featureVec

def getAvgFeatureVecs(tweets, model, num_features):
    counter = 0
    tweetFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")
#    all_tweets = len(tweets)
    for i, tweet in enumerate(tweets):
#       # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(tweets)))
            
        tweetFeatureVecs[counter] = featureVecMethod(tweet, model, num_features)
        counter = counter+1      
    return tweetFeatureVecs

#Calculate the vector matrix
X = getAvgFeatureVecs(text_data, model, num_features)
    
print ("========================")

from sklearn.cluster import KMeans
#make the k-means model
NUM_CLUSTERS=20
kmeans = KMeans(NUM_CLUSTERS, random_state=0, max_iter=100, n_init=1, verbose=True).fit_predict(X)
labels = list(kmeans)

#make a dictionary with tweets and labels
values = labels
keys = tweets_text
KMeans_clusters = dict(zip(keys, values))

import csv
#write a table 
file_name = "twitter_kmeans_clusters.csv"
dictionary = KMeans_clusters
row1 = "tweet"
row2 = "cluster"

def writeCsv(file_name, row1, row2, dictionary):
   
    with open(file_name,'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow((row1, row2))
        for key, value in dictionary.items():
            writer.writerow([key, value])
            
writeCsv(file_name, row1, row2, dictionary)


