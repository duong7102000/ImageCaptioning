from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization, GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.callbacks import History

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# Load the efficient net


model = EfficientNetB0(weights='imagenet')
Embedding_CV_Layer = Model(model.input, model.layers[-2].output)
# size of efficentNet
in_size_CV = model.input.shape[1]
out_size_CV = model.layers[-2].output.shape[1]

# Build NLP embedding layer
Embedding_NLP_Layer = Embedding(vocab_size, embedding_dim, mask_zero=True)
input_NLP_layer = Input(shape=(max_length,))
output_NLP_layer = Embedding_NLP_Layer(input_NLP_layer)
Embedding_NLP_Layer = Model(input_NLP_layer, output_NLP_layer)
# Embedding_NLP_Layer.set_weights([embedding_matrix])
Embedding_NLP_Layer.layers[1].set_weights([embedding_matrix])
Embedding_NLP_Layer.layers[1].trainable = False
Embedding_NLP_Layer.summary()


# Tạo model
class CV_warmup_layer(layers.Layer):
    def __init__(self):
        super(CV_warmup_layer, self).__init__()
        self.ff1 = Dense(1024, activation='relu')
        self.dropout = Dropout(0.3)
        self.ffout = Dense(512, activation='relu')

    def call(self, inputs):
        x = self.dropout(inputs)
        x = self.ff1(x)
        x = self.dropout(x)
        x = self.ffout(x)
        return x


# Layer CV
cv_warmup_layer = CV_warmup_layer()


# inputCV = Input(shape=(1792,))
# output_CV = cv_warmup_layer(inputCV)
# cv_warmup_layer = Model(inputCV, output_CV)

class NLP_layer(layers.Layer):
    def __init__(self):
        super(NLP_layer, self).__init__()
        self.droput = Dropout(0.5)
        self.LSTM = Bidirectional(GRU(256))
        # self.embed_layer = Embedding(vocab_size, embedding_dim, mask_zero=True).set_weights([embedding_matrix])
        self.embed_layer = Embedding_NLP_Layer

    def call(self, inputs):
        se1 = self.embed_layer(inputs)
        se2 = self.droput(se1)
        se3 = self.LSTM(se2)
        # print(se3)
        return se3


# Layer NLP
nlp_warmup_layer = NLP_layer()


# inputNLP = Input(shape=(max_length,))
# output_NLP = nlp_warmup_layer(inputNLP)
# nlp_warmup_layer = Model(inputNLP, output_NLP)

class Head(layers.Layer):
    def __init__(self):
        super(Head, self).__init__()
        self.ff1 = Dense(512, activation='relu')
        self.dropout = Dropout(0.3)
        self.ff2 = Dense(256, activation='relu')
        self.ffout = Dense(vocab_size, activation='softmax')

    def call(self, output_CV_NLP):
        decoder1 = add(output_CV_NLP)
        decoder2 = self.ff1(decoder1)
        decoder3 = self.dropout(decoder2)
        decoder4 = self.ff2(decoder3)
        outputs = self.ffout(decoder4)
        return outputs


# Layer Head
head = Head()


# inputCVHead = Input(shape=(512,))
# inputNLPHead = Input(shape=(512,))
# output_Head = head([inputCVHead, inputNLPHead])
# head = Model(inputs= [inputCVHead, inputNLPHead], outputs= output_Head)

class ICModel():
    def __init__(self, out_size_CV, use_embCV=False, input_size=False):
        self.cv_layer = cv_warmup_layer
        self.out_size_CV = out_size_CV
        self.nlp_layer = nlp_warmup_layer
        self.head = head
        self.use_embCV = use_embCV

        if use_embCV:
            self.embed_layer = Embedding_CV_Layer
            self.input_size = input_size

    def build_warmUpModel(self):
        if self.use_embCV:
            inputs1 = Input(shape=(self.input_size, self.input_size, 3))
            x = self.embed_layer(inputs1)
            x = Reshape((self.out_size_CV,))(x)
            output_CV = self.cv_layer(x)
        else:
            inputs1 = Input(shape=(self.out_size_CV,))
            output_CV = self.cv_layer(inputs1)

        inputs2 = Input(shape=(max_length,))
        output_NLP = self.nlp_layer(inputs2)

        outputs = self.head([output_CV, output_NLP])

        warmUpModel = Model(inputs=[inputs1, inputs2], outputs=outputs)
        return warmUpModel