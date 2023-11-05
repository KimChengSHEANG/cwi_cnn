#!/usr/bin/env bash
#python train.py

#python train.py --train_path='./data/CWI 2018 Training Set/english/News_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/News_Dev.tsv'

#python train.py --train_path='./data/CWI 2018 Training Set/english/WikiNews_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/WikiNews_Dev.tsv'

#python train.py --train_path='./data/CWI 2018 Training Set/english/All_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/All_Dev.tsv'


python trains.py --train_path='./data/CWI 2018 Training Set/english/Wikipedia_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/Wikipedia_Dev.tsv'