#!/usr/bin/env bash

python generate_embeddings.py -d ~/upf/11_resources/dataset/glove/glove.6B.300d.txt --npy_output ./data/dumps/embeddings.npy --dict_output ./data/dumps/vocab.pckl --dict_whitelist ./data/embeddings/vocab.txt
