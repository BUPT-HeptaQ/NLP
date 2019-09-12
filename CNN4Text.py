# 1 dimension for text
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb

# set parameters
max_features = 5000
max_len = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2
np.random.seed(1337)  # for reproducibility
# use the self-existent data set, load IMDB data
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
# this data set is the done BoW, and we make them into the same length, not enough -> 0, too much -> cut
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# initial our sequential model(refer to linear layers)
model = Sequential()

# here need a Embedding Layer to transfer input word indexes into tensor vectors,
# it's like the results of word2vec, but this Embedding has no big deal, just quantization the input
model.add(Embedding(max_features, embedding_dims, input_length=max_len, dropout=0.2))
# it is only useful for this BoW, for our data set, we need handle more details

# now we could add a new Convolution layer
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, boder_mode='valid',
                        activation='relu', subsample_length=1))
# follow a Maxpooling, the results are like butch of small vec, flatten them all
model.add(MaxPooling1D(pool_length=model.output_shape[1]))
model.add(Flatten())

# using Dense to express common Layer in Keras
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('sigmoid'))

# the last layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

# if we face to time sequence, here we could use LSTM to replace layers
# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)

