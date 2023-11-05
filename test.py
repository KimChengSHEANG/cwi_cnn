
import tensorflow as tf
import numpy as np
import os
import datetime
import time

from cwi_cnn import CWI_CNN
import data_helpers as utils
# from configure import FLAGS
import codecs
import os

import numpy as np
import pickle as pckl
import codecs

from data_helpers import *
from nltk.tokenize import word_tokenize

porter = PorterStemmer()
wnl = WordNetLemmatizer()


def process_line(sentence, tokenizer=word_tokenize):
    lemmas = []
    words = tokenizer(sentence)
    for word in words:
        lemmas.append(lemmatize(word.lower()))
        # lemmas.append(word.lower())

    lemmas = [lemma for lemma in lemmas if lemma.isalpha()]
    return lemmas


def lemmatize(word, lemmatizer=wnl, stemmer=porter):
    lemma = lemmatizer.lemmatize(word)
    stem = stemmer.stem(word)

    if not wordnet.synsets(lemma):
        if not wordnet.synsets(stem):
            return word
        else:
            return stem
    else:
        return lemma

# TRAIN_PATH = './data/CWI 2018 Training Set/english/News_Test.tsv'
TRAIN_PATH_News = './data/CWI 2018 Training Set/english/News_Train.tsv'
# DEV_PATH = './data/CWI 2018 Training Set/english/All_Dev.tsv'
TRAIN_PATH_WikiNews = './data/CWI 2018 Training Set/english/WikiNews_Train.tsv'
TRAIN_PATH_Wikipedia = './data/CWI 2018 Training Set/english/Wikipedia_train.tsv'

def extract_unique_words(filename):
    with open(filename) as data_file:
        lines = [line.split('\t') for line in data_file.read().splitlines()]
    print("#lines: ", len(lines))
    allwords = []
    for line in lines:
        words = process_line(line[1])
        allwords.extend(words)

    uwords = set(allwords)
    return uwords


# uwords = extract_unique_words(TRAIN_PATH_WikiNews)
# print(TRAIN_PATH_WikiNews)
# print(uwords)
# print(len(uwords))
#
#
# uwords = extract_unique_words(TRAIN_PATH_News)
# print(TRAIN_PATH_News)
# print(uwords)
# print(len(uwords))


uwords = extract_unique_words(TRAIN_PATH_Wikipedia)
print(TRAIN_PATH_Wikipedia)
print(uwords)
print(len(uwords))

