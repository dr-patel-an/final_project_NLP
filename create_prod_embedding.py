# TODO:
# 1. Replacing a number with NUM string
# 2. Understand the layers bi-lstm with attention


import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from custom_embedding import *
from bs4 import BeautifulSoup
import sys
import os
os.chdir('/Users/tejinderpalsingh/Desktop/MIDS/w266/textClassifier-master')
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from nltk import tokenize
import multiprocessing
import json

config = {
    'word2vec': {
        'n': 10,                 # dimensions of word embeddings, also refer to size of hidden layer
        'epochs': 5,            # number of training epochs
        'learning_rate': 0.01,  # learning rate
        'out_layer': 11         # Number of output layers 0 to 10 inclusive
    },
    'max_dict_size': 80000,     # Maximum dictionary/vocabulary size
    'num_processes': 1 
}

prod_to_process = sys.argv[1]

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def get_factorized_data(df):
    reviews, labels, users, products, texts = [], [], [], [], []
    for idx in range(df.review.shape[0]):
        text = BeautifulSoup(df.review[idx])
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        labels.append(df.sentiment[idx])
        users.append(df.user[idx])
        products.append(df['product'][idx])
    return reviews, labels, users, products, texts


def get_prod_embedding(all_data, product, word_index, return_dict):
    w2v = word2vec(config)
    df_product = all_data[all_data['product'] == product]
    training_data = w2v.generate_training_data(df_product['review'], df_product['sentiment'], word_index)
    w2v.train(training_data)
    prod_embed = w2v.get_embedding_dict()
    del w2v
    return_dict["{}".format(product)] = prod_embed


# Reading and factorizing training and testing data
data_train = pd.read_csv('imdb/train_small.txt', sep='\t')
#data_train = data_train[0:100]
data_train.columns = ["unname","user", "product", "sentiment", "review"]
train_reviews, train_labels, train_users, train_products, train_texts = get_factorized_data(data_train)

data_val = pd.read_csv('imdb/dev_small.txt', sep='\t')
#data_val = data_val[0:20]
data_val.columns = ["unname","user", "product", "sentiment", "review"]
val_reviews, val_labels, val_users, val_products, val_texts = get_factorized_data(data_val)


all_data = data_train.append(data_val, ignore_index=True)

del data_val
del data_train


## Apply glove embedding to both training and test data
tokenizer = Tokenizer(nb_words=config['max_dict_size'])
tokenizer.fit_on_texts(train_texts + val_texts )
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))



## checking if embedding for prod done 

path_to_json ='/Volumes/External/product_embedding'

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

print(json_files)

prod_list_done=[]

for j in json_files:
    with open('{}/{}'.format(path_to_json, j)) as i:
         prod_val=list(json.load(i).keys())[0]
         prod_list_done.append(prod_val)

# Applying product embedding
# read prod files to process 
f = open(prod_to_process,'r')
line = f.readline()
all_prods_pre=line.split(",")
#all_prods_pre = list(set(all_data['product']))
# remove already done prods
all_prods = [item for item in all_prods_pre if item not in prod_list_done]
del all_prods_pre


num_process = config['num_processes']
segment_prods = [all_prods[i * num_process : (i + 1) * num_process] for i in range((len(all_prods) + num_process - 1) // num_process)]
manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []

## id based on what have been processed so far
id_numbers=[]
for i in json_files:
    id_numbers.append(int(i.split("_")[2].split(".")[0]))

id = max(id_numbers)  

del json_files

for prods in segment_prods:
    id += 1
    return_dict = manager.dict()
    for prod in prods:
        p = multiprocessing.Process(target=get_prod_embedding, args=(all_data, prod, word_index, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    with open("/Volumes/External/product_embedding/{}_embedding_{}.json".format(prod_to_process,id), 'w') as outfile:
        json.dump(return_dict.copy(), outfile)
     
    del return_dict
print("Done!")






