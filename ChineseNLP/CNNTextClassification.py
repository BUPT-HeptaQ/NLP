# use deep learning CNN to make text classification
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jieba
import pandas as pd
import argparse
import sys
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

data_file_technology = pd.read_csv("D:technology_news.csv", encoding='utf-8')
data_file_technology = data_file_technology.dropna()

data_file_car = pd.read_csv("D:car_news.csv", encoding='utf-8')
data_file_car = data_file_car.dropna()

data_file_entertainment = pd.read_csv("D:entertainment_news.csv", encoding='utf-8')
data_file_entertainment = data_file_entertainment.dropna()

data_file_military = pd.read_csv("D:military_news.csv", encoding='utf-8')
data_file_military = data_file_military.dropna()

data_file_sports = pd.read_csv("D:sports_news.csv", encoding='utf-8')
data_file_sports = data_file_sports.dropna()

technology = data_file_technology.content.values.tolist()[1000:21000]
car = data_file_car.content.values.tolist()[1000:21000]
entertainment = data_file_entertainment.content.values.tolist()[2000:22000]
military = data_file_military.content.values.tolist()[2000:22000]
sports = data_file_sports.content.values.tolist()[:20000]

# load the stopwords
stopwords = pd.read_csv("D:stopwords.txt", index_col=False, quoting=3,
                        sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values


# preprocessing data
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segments = jieba.lcut(line)
            segments = filter(lambda x: len(x) > 1, segments)
            segments = filter(lambda x: x not in stopwords, segments)
            sentences.append(" ".join(segments), category)

        except (OSError, TypeError) as reason:
            print("the error info is:", str(reason))
            print(line)
            continue


# generate unsupervised train data
sentences = []

preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')

# Divide the original data set into test sets of training sets, using sklearn's own segmentation function "Zip!"
text, label = zip(*sentences)
train_data, test_data, train_target, test_target = train_test_split(text, label, random_state=1234)


""" use TensorFlow to build Chinese text classification based on convolutional neural network """

learn = tf.contrib.learn

FLAGS = None

# the longest length of document
MAX_DOCUMENT_LENGTH = 100
# the minimum frequency of words
MIN_WORD_FREQUENCY = 2
# the word embedding dimension
EMBEDDING_SIZE = 20
# the number of filters
N_FILTERS = 10
# window size
WINDOW_SIZE = 20
# the shape of filters
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
# pooling (pooling will loss some in data sampling processing and effect accuracy, or ignore this step)
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0


def CNN_model(features, target):
    """
     two layers of convolutional neural network, used to short text classification
    """
    # turn words into embedding
    # get a vocabulary map to maxtrix, and its shape is [n_words, EMBEDDING_SIZE]
    # Map a batch of text to a matrix of [batch_size, sequence_length, EMBEDDING_SIZE]
    target = tf.one_hot(target, 5, 1, 0)
    word_vectors = tf.contrib.laysers.embed_sequence(
        features, vocab_size=n_words, emded_sequence=EMBEDDING_SIZE, scope='words')
    word_vectors = tf.expand_dims(word_vectors, 3)

    with tf.variable_scope('CNN_Layer1'):
        # add convolutional layer to be the filter
        convolution1 = tf.contrib.layer.convolution2d(
            word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # add RELU non linear
        convolution1 = tf.nn.relu(convolution1)

        # max pooling
        pooling1 = tf.nn.max_pool(convolution1, ksize=[1, POOLING_WINDOW, 1, 1],
                                  strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Invert the matrix to satisfy the shape
        pooling1 = tf.transpose(pooling1, [0, 1, 3, 2])

    with tf.variable_scope('CNN_Layer2'):
        # the second convolutional layer
        convolution2 = tf.contrib.layer.convolution2d(
            pooling1, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # get the features
        pooling2 = tf.squeeze(tf.reduce_max(convolution2, 1), squeeze_dims=[1])

    # all connection layer
    logits = tf.contrib.layers.fully_connected(pooling2, 5, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    train_operation = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.01)

    return ({'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_operation)


"""
# build data set
text_train = pd.DataFrame(train_data)[1]
label_train = pd.Series(train_target)
text_test = pd.DataFrame(test_data)[1]
label_test = pd.Series(test_target)
"""


# learn has the VocabularyProcessor in tensorflow.preprocessing package
tmp = ['I am good', 'you are here', 'I am glad', 'it is great']
vocab_processor = learn.preprocessing.VocabularyPreprocessor(10, min_frequency=1)
list(vocab_processor.fit_tr)

global n_words
# process the vocabulary
vocab_processor = learn.preprocessing.VocabularyPreprocessor(MAX_DOCUMENT_LENGTH, mini_frenquency=MIN_WORD_FREQUENCY)
text_train = np.array(list(vocab_processor.fit_transform(train_data)))
text_test = np.array(list(vocab_processor.fit_transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)
# Total words: 50281

category_dictionary = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}
train_target = map(lambda x: category_dictionary[x], train_target)
test_target = map(lambda x: category_dictionary[x], test_target)
label_train = pd.Series(train_target)
label_test = pd.Series(test_target)

# build model
classifier = learn.SKCompat(learn.Estimator(model_fn=CNN_model))

# train and prediction
classifier.fit(text_train, label_train, steps=1000)  # improved accuracy needs more iterate steps
label_predicted = classifier.predict(text_test)['class']
score = metrics.accuracy_score(label_test, label_predicted)
print('Accuracy: {0:f}'.format(score))
