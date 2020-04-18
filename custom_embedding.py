from collections import defaultdict
import numpy as np
import pdb
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical


class word2vec():
    def __init__(self, settings):
	np.random.seed(1000)
        self.n = settings['word2vec']['n']
        self.lr = settings['word2vec']['learning_rate']
        self.epochs = settings['word2vec']['epochs']
        self.out_layer = settings['word2vec']['out_layer']
        self.v_count = 0
        self.words_list = []
        self.word_index = dict()
        self.index_word = dict()

    def generate_training_data(self, text_list, polarity, word_index):
        # Find unique word counts using dictionary
        all_words = word_index.keys()
        word_counts = defaultdict(int)
        polarity = list(polarity)
        corpus = [[word.lower() for word in text.split()] for text in text_list]
        for row in corpus:
            for word in row:
                if word in all_words:
                    word_counts[word] += 1

        # Generate word:index
        self.word_index = word_index
        # How many unique words in vocab? 9
        self.v_count = len(self.word_index.keys())
        # Generate Lookup Dictionaries (vocab)
        self.words_list = all_words

        # Generate index:word
        self.index_word = dict()
        for word in self.word_index:
            self.index_word[self.word_index[word]] = word

        training_data = []
        # Cycle through each sentence in corpus
        for inx, sentence in enumerate(corpus):
            sent_len = len(sentence)
            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot; 9x1 vector
                if sentence[i] in all_words:
                    w_target = self.word2onehot(sentence[i])
                    # Cycle through context window; 11x1 vector
                    w_context = [0 for _ in range(self.out_layer)]
                    w_context[polarity[inx]] = 1
                    training_data.append([w_target, w_context])
        return np.array(training_data)

    def word2onehot(self, word):
        # word_vec - initialise a blank vector
        word_vec = [0 for _ in range(0, self.v_count)]
        # Get ID of word from word_index
        word_inx = self.word_index[word]
        # Change value from 0 to 1 according to ID of the word
        word_vec[word_inx-1] = 1        # Since words are assigned indices starting from 1, self.v_count+1
        return word_vec

    def train(self, training_data):
        # Initialising weight matrices
        # Both s1 and s2 should be randomly initialised but for this demo, we pre-determine the arrays (getW1 and getW2)
        # w1 - shape (9x1) and w2 - shape (1x11), where 11 is the max number of polarities
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.out_layer))

        # Cycle through each epoch
        for i in range(self.epochs):
            # Intialise loss to 0
            self.loss = 0
            # Cycle through each training sample
            # w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
                # Forward pass - Pass in vector for target word (w_t) to get:
                # 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
                y_pred, h, u = self.forward_pass(w_t)

                # Calculate error
                # 1. For a target word, calculate difference between y_pred and each of the context words
                # 2. Sum up the differences using np.sum to give us the error for this particular target word
                EI = np.subtract(y_pred, w_c)

                # Backpropagation
                # We use SGD to backpropagate errors - calculate loss on the output layer
                self.backprop(EI, h, w_t)

                # Calculate loss
                # There are 2 parts to the loss function
                # Part 1: -ve sum of all the output +
                # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
                # Note: word.index(1) returns the index in the context word vector with value 1
                # Note: u[word.index(1)] returns the value of the output layer before softmax
                # self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                self.loss += -np.sum(np.dot(w_c, np.log2(y_pred)))
            print('Epoch:', i, "Loss:", self.loss)

    def forward_pass(self, x):
        # x is one-hot vector for target word, shape - 9x1
        # Run through first matrix (w1) to get hidden layer - 1x9 dot 9x1 gives us 1x1
        h = np.dot(self.w1.T, x)
        # Dot product hidden layer with second matrix (w2) - 11x1 dot 1x1 gives us 11x1
        u = np.dot(self.w2.T, h)
        # Run 1x11 through softmax to force each element to range of [0, 1] -
        y_c = self.softmax(u)
        return y_c, h, u

    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 1x1, e - shape 11x1, dl_dw2 - shape 1x11
        dl_dw2 = np.outer(h, e)
        # x - shape 9x1, w2 - 1x11, e.T - 1x11
        # x - 9x1, np.dot() - 1x1, dl_dw1 - 9x1
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def get_embedding_dict(self):
        embedding_dict = dict()
        for word in self.word_index:
            w_index = self.word_index[word]
            embedding_dict[word] = list(self.w1[w_index-1])    # Since the index starts with 1
        return embedding_dict

    def get_embedding(self):
        return self.w1

    def get_word_dict(self):
        return self.word_index
