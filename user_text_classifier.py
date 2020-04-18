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
import json

config = {
    'word2vec': {
        'n': 10,                 # dimensions of word embeddings, also refer to size of hidden layer
        'epochs': 3,            # number of training epochs
        'learning_rate': 0.01,  # learning rate
        'out_layer': 11         # Number of output layers 0 to 10 inclusive
    },
    'max_sent_length':  100,     # Number of words in a sentence
    'max_sent': 15,             # Maximum number of sentences in a review
    'max_dict_size': 80000,     # Maximum dictionary/vocabulary size
    'embedding_dim': 100,       # Embedding dimension
    'polarities': 11            # Number of classes including zero
}


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


# Reading and factorizing training and testing data
data_train = pd.read_csv('./imdb/train_small.txt', sep='\t')
#data_train = data_train[0:100]
data_train.columns = ["unnamed", "user", "product", "sentiment", "review"]
train_reviews, train_labels, train_users, train_products, train_texts = get_factorized_data(data_train)

data_val = pd.read_csv('./imdb/dev_small.txt', sep='\t')
#data_val = data_val[0:20]
data_val.columns = ["unnamed", "user", "product", "sentiment", "review"]
val_reviews, val_labels, val_users, val_products, val_texts = get_factorized_data(data_val)

# data_test = pd.read_csv('./imdb/test_original.txt', sep='\t\t')
# data_test = data_test[0:20]
# data_test.columns = ["user", "product", "sentiment", "review"]
# test_reviews, test_labels, test_users, test_products, test_texts = get_factorized_data(data_test)

all_data = data_train.append(data_val, ignore_index=True)
#partial_data = data_train.append(data_val, ignore_index=True)
#all_data = partial_data.append(data_test, ignore_index=True)

# Apply glove embedding to both training and test data
tokenizer = Tokenizer(nb_words=config['max_dict_size'])
tokenizer.fit_on_texts(train_texts + val_texts)
#tokenizer.fit_on_texts(train_texts + val_texts + test_texts)
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

# Applying user embedding
all_users = list(set(all_data['user']))
all_prods = list(set(all_data['product']))

files = []
user_embedding = dict()

path = './user_embedding/userset1'
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.json' in file:
            files.append(os.path.join(r, file))


for file in files:
    with open(file) as json_file:
        user_embedding.update(json.load(json_file))

path = './user_embedding/userset2'
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.json' in file:
            files.append(os.path.join(r, file))


for file in files:
    with open(file) as json_file:
        user_embedding.update(json.load(json_file))

# Applying product embedding
path = './prod_embedding'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.json' in file:
            files.append(os.path.join(r, file))

prod_embedding = dict()
for file in files:
    with open(file) as json_file:
        prod_embedding.update(json.load(json_file))

# Get user_index and prod_index
user_index = dict((user, i) for i, user in enumerate(set(all_users)))
index_user = dict((i, user) for i, user in enumerate(set(all_users)))

prod_index = dict((prod, i) for i, prod in enumerate(set(all_prods)))
index_prod = dict((i, prod) for i, prod in enumerate(set(all_prods)))

GLOVE_DIR = "./glove/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors.' % len(embeddings_index))

num_users, num_prods, num_words = len(user_index), len(prod_index), len(word_index)
user_prod_word_dim = ((num_users+1)*(num_prods+1)*(num_words+1))
enhanced_embedding_dim = (config['embedding_dim'] + (2 * config['word2vec']['n']))
# Added one for 'UNK' word and (2 * config['word2vec']['n']) added for user and product embedding
embedding_matrix = np.random.random((user_prod_word_dim, enhanced_embedding_dim))

for user, i in user_index.items():
    for prod, j in prod_index.items():
        for word, k in word_index.items():
            embedding_vector, user_vector, prod_vector = None, None, None
            inx = (i * (num_prods + 1) * (num_words + 1)) + (j * (num_words + 1)) + k
            embedding_vector = embeddings_index.get(word)
            embed_user = user_embedding.get(user)
            if embed_user is not None:
                user_vector = embed_user.get(word)
            embed_prod = prod_embedding.get(prod)
            if embed_prod is not None:
                prod_vector = embed_prod.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[inx][0:config['embedding_dim']] = embedding_vector
            if user_vector is not None:
                embedding_matrix[inx][config['embedding_dim']:(config['embedding_dim'] + config['word2vec']['n'])] = user_vector
            if prod_vector is not None:
                embedding_matrix[inx][(config['embedding_dim'] + config['word2vec']['n']):(config['embedding_dim'] + (2*config['word2vec']['n']))] = prod_vector


embedding_layer = Embedding(user_prod_word_dim,
                            enhanced_embedding_dim,
                            weights=[embedding_matrix],
                            input_length=config['max_sent_length'],
                            trainable=True,
                            mask_zero=True)

train_encoded_data = np.zeros((len(train_texts), config['max_sent'], config['max_sent_length']), dtype='int32')

for i, sentences in enumerate(train_reviews):
    for j, sent in enumerate(sentences):
        if j < config['max_sent']:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < config['max_sent_length'] and tokenizer.word_index[word] < config['max_dict_size']:
                    inx = (user_index[train_users[i]] + (num_prods+1) + (num_words+1)) + \
                          (prod_index[train_products[i]] * (num_words+1)) + \
                          tokenizer.word_index[word]
                    train_encoded_data[i, j, k] = inx
                    k = k + 1

train_encoded_labels = to_categorical(np.asarray(train_labels))
print('Shape of train data tensor:', train_encoded_data.shape)
print('Shape of train label tensor:', train_encoded_labels.shape)

val_encoded_data = np.zeros((len(val_texts), config['max_sent'], config['max_sent_length']), dtype='int32')

for i, sentences in enumerate(val_reviews):
    for j, sent in enumerate(sentences):
        if j < config['max_sent']:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < config['max_sent_length'] and tokenizer.word_index[word] < config['max_dict_size']:
                    inx = (user_index[val_users[i]] + (num_prods + 1) + (num_words + 1)) + \
                          (prod_index[val_products[i]] * (num_words + 1)) + \
                          tokenizer.word_index[word]
                    val_encoded_data[i, j, k] = inx
                    k = k + 1

val_encoded_labels = to_categorical(np.asarray(val_labels))
print('Shape of train data tensor:', val_encoded_data.shape)
print('Shape of train label tensor:', val_encoded_labels.shape)

x_train = train_encoded_data
y_train = train_encoded_labels
x_val = val_encoded_data
y_val = val_encoded_labels

print('Number of positive and negative reviews in training and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

gru_seq_len = config['max_sent_length']

sentence_input = Input(shape=(config['max_sent_length'],), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(gru_seq_len, return_sequences=True))(embedded_sequences)
l_att = AttLayer(gru_seq_len)(l_lstm)
sentEncoder = Model(sentence_input, l_att)
#sentEncoder = Model(sentence_input, l_lstm)

print(sentEncoder.summary())

gru_seq_len_layer2 = config['max_sent']

review_input = Input(shape=(config['max_sent'], config['max_sent_length']), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(gru_seq_len_layer2, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(gru_seq_len_layer2)(l_lstm_sent)
preds = Dense(config['polarities'], activation='softmax')(l_att_sent)
#preds = Dense(config['polarities'], activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)

print(model.summary())
