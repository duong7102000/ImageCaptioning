# Thêm thư viện
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
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow
from keras.callbacks import History
from tensorflow.keras.applications import EfficientNetB0
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class Preprocess():
    def __init__(self):
        model = EfficientNetB0(weights='imagenet')
        image_embeddings = Model(model.input, model.layers[-2].output)
        image_embeddings.trainable = False

    def preprocess(self, image_path):
        # Convert all the images to size 299x299 as expected by the inception v3 model
        img = image.load_img(image_path, target_size=(380, 380))
        # Convert PIL image to numpy array of 3-dimensions
        x = image.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        x = tensorflow.keras.applications.efficientnet.preprocess_input(x)
        return x
