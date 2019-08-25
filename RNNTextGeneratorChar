import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load the text file
raw_text = open(' ').read()
raw_text = raw_text.lower()

# use One-Hot to encode all chars
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# build the train data set
# transfer our raw data into practical x, y; x are preposition chars, y is the next one
seq_length = 100
x = []
y = []
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

'''
1. we have already an input digital index, we need out them into LSTM array format[number of sample: time batch: features]
2. we use one-hot to predict the output, compare to predict a exact y, it's much better  
'''

n_patterns = len(x)
n_vocab = len(chars)
# change x into LSTM pattern
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# normalize between 0-1
x = x / float(n_vocab)
# output turn into one-hot
y = np_utils.to_categorical(y)

# build the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x, y, nb_epoch=50, batch_size=4096)

# to show the outputs of the LSTM
def predict_next(input_arary):
    x = numpy.reshape(input_arary, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input) - seq_length):]:
        res.append(char_to_int[c])
    return res

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c

def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = ' '
article = generate_article(init)
print(article)


