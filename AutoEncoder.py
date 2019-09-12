from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cluster import KMeans

class ASCII_Auto_encoder():

    def __init__(self, sen_len=512, encoding_dim=32, epoch=50, val_ratio=0.3):
        '''
        this encoder is based on ASCII characters
        :param sen_len: put the sentence pad as the same length
        :param encoding_dim: the dimensions after compressed
        :param epoch: note how many epochs has to implement
        :param val_ratio: simple KNN clustering model
        '''
        self.sen_len = sen_len
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.kmeanmodel = KMeans(n_clusters=2)
        self.epoch = epoch

    def fit(self, x):
        '''
        build the model
        :param x:  input text
        '''

     # make all trainsets into same size(cut or extend into 512), and transfer every characters into ASCII code
        x_train = self.preprocess(x, length=self.sen_len)
     # leave the input position
        input_text = Input(shape=(self.sen_len, ))

     # "encoded" go through each layer, will be refreshed into smaller "compressed impression"
        encoded = Dense(1024, activation='tanh')(input_text)
        encoded = Dense(512, activation='tanh')(encoded)
        encoded = Dense(128, activation='tanh')(encoded)
        encoded = Dense(self.encoding_dim, activation='tanh')(encoded)

     # "decoded" is reverse the compressed trainsets turn back into input_text
        decoded = Dense(128, activation='tanh')(encoded)
        decoded = Dense(512, activation='tanh')(decoded)
        decoded = Dense(1024, activation='tanh')(decoded)
        decoded = Dense(self.sen_len, activation='sigmoid')(decoded)

     # this auto encoder model
        self.autoencoder = Model(input=input_text, output=decoded)
     # the half model is encoder
        self.encoder = Model(input=input_text, output=encoded)

     # leave the position fpr input size of encoded
        encoded_input = Input(shape=(1024,))
     # the last layer of the autoencoder is the first layer of decoder
        decoder_layer = self.autoencoder.layers[-1]
     # connect the start and end, it is a decoder
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
     #compile
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(x_train, x_train, nb_epoch=self.epoch, batch_size=1000, shuffle=True, )

     # this part use KNN to train yourself, a simple classifier which based on distance
        x_train = self.encoder.predict(x_train)
        self.kmeanmodel.fit(x_train)

    def predict(self, x):
        """
        make predict
        :param x: input text
        :return: predictions
        """
     # the first step, make all data transfer into ASCII, and have the same length:512
        x_test = self.preprocess(x, length=self.sen_len)
     # use decoder to compress the test dataset
        x_test = self.encoder.predict(x_test)
     # use KNN to claasify data
        preds = self.kmeanmodel.predict(x_test)

        return preds

    def preprocess(self, s_list, length=256):
        


