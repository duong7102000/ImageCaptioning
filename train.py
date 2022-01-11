import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
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
import os

from model import ICModel, out_size_CV, in_size_CV


class Train:
    def __init__(self):
        self.train_features_finetune = load(open("image_pkl/encoded_train_images.pkl", "rb"))
        self.weight_dir = 'model_weights/model_weights/finetune/weight_EB0_biLSTM_21-12-2021-01_38_44.h5'
        self.FinetuneModel = ICModel(out_size_CV, use_embCV=True, input_size=in_size_CV)
        self. FinetuneModel = self.FinetuneModel.build_warmUpModel()
        self.weights_finetune = os.listdir(self.weight_dir)
        # FinetuneModel = FinetuneModel.build_warmUpModel()
        if len(self.weights_finetune):
            self.FinetuneModel.load_weights(os.path.join(self.weight_dir, self.weights_finetune[-1]))
        self.FinetuneModel.summary()
        self.FinetuneModel.compile(loss='categorical_crossentropy', optimizer='adam')
        self.FinetuneModel.optimizer.lr = 0.001
        Finetune_number_pics_per_bath = 1
        Finetune_steps = len(self.train_descriptions) // Finetune_number_pics_per_bath

    def data_generator_finetune(self, descriptions, photos, wordtoix, max_length, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n=0
        # loop for ever over images
        while 1:
            for key, desc_list in descriptions.items():
                n+=1
                # retrieve the photo feature
                photo = photos[key+'.jpg']
                for desc in desc_list:
                    # encode the sequence
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    # split one sequence into multiple X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        # store
                        X1.append(photo.reshape((in_size_CV,in_size_CV,3)))
                        X2.append(in_seq)
                        y.append(out_seq)
                # yield the batch data
                if n==num_photos_per_batch:
                    yield ([array(X1), array(X2)], array(y))
                    X1, X2, y = list(), list(), list()
                    n=0

    def train(self):
        self.FinetuneModel.compile(loss='categorical_crossentropy', optimizer='adam')
        self.FinetuneModel.optimizer.lr = 0.001
        Finetune_number_pics_per_bath = 1
        Finetune_steps = len(self.train_descriptions) // Finetune_number_pics_per_bath
        train_finetune_generator = self.data_generator_finetune(self.train_descriptions, self.train_features_finetune, self.wordtoix,
                                                           self.max_length, Finetune_number_pics_per_bath)
        # val_finetune_generator = data_generator_finetune(validation_descriptions, validation_features_finetune, wordtoix, max_length, Finetune_number_pics_per_bath)

        FinetuneHistory = self.FinetuneModel.fit_generator(train_finetune_generator, epochs=2, verbose=1, \
                                                      callbacks=self.call_backs('finetune'),
                                                      steps_per_epoch=Finetune_steps)
        self.FinetuneModel.save_weights('/model_weights/bsl_e20_b60.h5')
