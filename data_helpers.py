import os

import numpy as np
import pickle as pckl

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from configure import *

porter = PorterStemmer()
wnl = WordNetLemmatizer()

params = DefaultConfiguration()


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


def generate_vocab(filenames, write_filename='./data/embeddings/vocab.txt'):
    vocabulary = []
    for filename in filenames:
        with open(filename) as data_file:
            lines = [line.split('\t') for line in data_file.read().splitlines()]

        for line in lines:
            words = process_line(line[1])
            vocabulary += [word for word in words if word not in vocabulary]

    vocabulary.sort()

    with open(write_filename, 'w') as write_file:
        for word in vocabulary:
            write_file.write("%s\n" % word)
    return


def map_to_vocab(sentence, vocab_dict='./data/dumps/vocab.pckl'):
    with open(vocab_dict, 'rb') as vfile:
        v_dict = pckl.load(vfile)

    return [v_dict.get(word, 0) for word in sentence]


def load_vocab_size():
    vocab_file = './data/embeddings/vocab.txt'
    with open(vocab_file) as data_file:
        lines = [line for line in data_file.read().splitlines()]
        return len(lines)
    return 0


def zero_pad(sequence, max_len=600):
    # sequence length is approx. equal to
    # the max length of sequence in train set
    return np.pad(sequence, (0, max_len - len(sequence)), mode='constant') \
        if len(sequence) < max_len else np.array(sequence[:max_len])


def get_word_vector(word, embedding_matrix):
    word_index = map_to_vocab([word])
    return embedding_matrix[word_index]


def get_context_words_vector_average(phrase, embedding_matrix):
    words = process_line(phrase)

    size = len(words)
    sum = np.zeros((1, params.TEXT_EMBEDDING_DIM))
    if size > 0:
        for word in words:
            word_vector = get_word_vector(word, embedding_matrix)
            sum += np.array(word_vector)
        return (sum/size)[0]
    else:
        return np.zeros(params.TEXT_EMBEDDING_DIM)


def get_target_word_vectors(target, embedding_matrix):

    words = process_line(target)
    target_word_vectors = np.zeros((params.MAX_TARGET_WORDS, params.TEXT_EMBEDDING_DIM))
    for i, word in enumerate(words):
        if i < params.MAX_TARGET_WORDS:
            target_word_vectors[i] = get_word_vector(word, embedding_matrix)

    return target_word_vectors


def process(line, embedding_matrix):
    sentence = line[1]
    target_offset_start = int(line[2])
    target_offset_end = int(line[3])
    target_phrase = line[4]
    left_context_phrase = sentence[:target_offset_start]
    right_context_phrase = sentence[target_offset_end:]

    left_context_vector = get_context_words_vector_average(left_context_phrase, embedding_matrix)
    right_context_vector = get_context_words_vector_average(right_context_phrase, embedding_matrix)

    target_word_vectors = get_target_word_vectors(target_phrase, embedding_matrix)
    # print(line)
    y = int(line[9])
    x = [left_context_vector, right_context_vector]
    x.extend(target_word_vectors)
    x = np.array(x)
    # x.extend(target_word_vectors)

    return x, y


def load_data_from_file(filename):
    embedding_path = './data/dumps/embeddings.npy'
    embedding_matrix = load_embeddings(embedding_path, load_vocab_size(), params.TEXT_EMBEDDING_DIM)

    with open(filename, 'r') as data_file:
        lines = [line.split('\t') for line in data_file.read().splitlines()]
    x = []
    y = []
    # i = 0
    for line in lines:
        _x, _y = process(line, embedding_matrix)
        x.append(_x)
        y.append(_y)
        # i += 1
        # if i == 100: break

    x = np.array([_x for _x in x])
    y = np.array([[0, 1] if _y == 1 else [1, 0] for _y in y])
    return x, y


def fetch(filename):
    """
    Fetch the preprocessed data from dump
    """
    x, y = pckl.load(open(filename, mode="rb"))
    return x, y


def create_dump(filename, write_filename):
    x, y = load_data_from_file(filename)
    pckl.dump((x, y), open(write_filename, "wb"))
    return


def load_embeddings(path, size, dimensions):
    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx + chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix


def batch_iter(data, batch_size, n_epochs, shuffle=False):
    print("Generating batch iterator ...")
    data = np.array(data)
    data_size = len(data)
    n_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    for epoch in range(n_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(n_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def write_configure_to_file(args, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fout = open(out_dir + "/configs.txt", "w")
    for arg in vars(args):
        fout.write("{} = {} \n".format(arg.upper(), getattr(args, arg)))
    fout.close()


if __name__ == '__main__':
    # word = "test"
    # print(word.isalpha())

    # extract Left Context
    filename = "data/english/All_Dev.tsv"
    # x, y = load_data_from_file(filename)
    # print(x)
    # print(x.shape)
    # print(x[0].shape)


