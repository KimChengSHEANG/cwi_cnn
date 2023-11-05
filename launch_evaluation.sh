#!/usr/bin/env bash
#python evaluate.py --ckptdir=`ls -lc runs | tail -1| cut -d ' ' -f13`


python evaluate.py --ckptdir=`ls runs | sort | tail -1` --dataset='./data/english/News_Test.tsv'
python evaluate.py --ckptdir=`ls runs | sort | tail -1` --dataset='./data/english/WikiNews_Test.tsv'
python evaluate.py --ckptdir=`ls runs | sort | tail -1` --dataset='./data/english/Wikipedia_Test.tsv'


