""" 
The long short term Memory (LSTM) neural network, which can capture temporal information,
has a strong ability to capture long information with its own Memory attributes 
"""

# use RNN to finish text classification

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
from tensorflow.contrib.layers.python.layers import encoders
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

learn = tf.contrib.learn

FLAGS = None

# the longest length of document
MAX_DOCUMENT_LENGTH = 15
# the minimum frequency of words
MIN_WORD_FREQUENCY = 1
# the word embedding dimension
EMBEDDING_SIZE = 50

global n_words
# process the vocabulary
vocab_processor = learn.preprocessing.VocabularyPreprocessor(MAX_DOCUMENT_LENGTH, mini_frenquency=MIN_WORD_FREQUENCY)
text_train = np.array(list(vocab_processor.fit_transform(train_data)))
text_test = np.array(list(vocab_processor.fit_transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)


def bag_of_words_model(features, target):
    """ first turn into bag of words model """
    target = tf.one_hot(target, 15, 1, 0)
    features = encoders.bow_encoder(features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    logits = tf.contrib.layers.fully_connected(features, 15, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    train_operation = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                                      optimizer='Adam', learning_rate=0.01)

    return({'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_operation)


category_dictionary = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}
train_target = map(lambda x: category_dictionary[x], train_target)
test_target = map(lambda x: category_dictionary[x], test_target)
label_train = pd.Series(train_target)
label_test = pd.Series(test_target)

model = bag_of_words_model
classifier = learn.SKCompat(learn.Estimator(model_fn=model))

# train and predict
classifier.fit(text_train, label_train, steps=1000)
label_predicted = classifier.predict(text_test)['class']
score = metrics.accuracy_score(label_test, label_predicted)
print('Accuracy: {0: f.}'.format(score))
