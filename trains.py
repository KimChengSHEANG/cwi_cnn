import tensorflow as tf
from train import *
from configure import *
from evaluate import evaluate
import os

LIST_LEARNING_RATES = [1e-3, 1e-2]
LIST_FILTER_SIZES = ["3,4,5", "2, 3,4"]
LIST_NUM_FILTERS = [32, 64, 128, 256]
LIST_DROPOUT_KEEP_PROB = [0.5, 0.75, 0.9]
LIST_L2_REG_LAMBDA = [1e-2, 1e-3, 1e-4]
LIST_BATCH_SIZE = [64, 128, 256]
LIST_NUM_EPOCHS = [50, 100]



def main(_):
    trains(LIST_LEARNING_RATES, LIST_FILTER_SIZES, LIST_NUM_FILTERS, LIST_DROPOUT_KEEP_PROB, LIST_L2_REG_LAMBDA, LIST_BATCH_SIZE, LIST_NUM_EPOCHS)


if __name__ == '__main__':
    tf.app.run()
