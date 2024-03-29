import os
import numpy as np
import nltk

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

raw_text = ' '
for file in os.listdir(" "):
    if file.endswith(" .txt"):
        raw_text += open(" " + file, errors='ignore').read() + '\n\n'

raw_text = raw_text.lower()
sentensor = nltk.data.load(' ')
sents = sentensor.tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))

print(len(corpus))

w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)

raw_input = [item for sublist in corpus for item in sublist]
len(raw_input)

text_stream = []
vocab = w2v_model.vocabulary
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
len(text_stream)

seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])

x = np.reshape(x, (-1, seq_length, 128))
y = np.reshape(y, (-1, 128))

model = Sequential()
model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, nb_epoch=50, batch_size=4096)

