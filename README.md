#CWI-NN: Complex Word Identification using Convolutional Neural Network and Feature Engineering

### About model

### Steps
Generate embeddings and vocabulary

```
    python generate_embeddings.py -d ~/upf/11_resources/dataset/glove/glove.6B.300d.txt --npy_output ./data/dumps/embeddings.npy --dict_output ./data/dumps/vocab.pckl --dict_whitelist ./data/embeddings/vocab.txt
    
```
###Train

```
python train.py --train_path='./data/CWI 2018 Training Set/english/News_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/News_Dev.tsv'
```

```
python train.py --train_path='./data/CWI 2018 Training Set/english/WikiNews_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/WikiNews_Dev.tsv'
```

```
python train.py --train_path='./data/CWI 2018 Training Set/english/Wikipedia_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/Wikipedia_Dev.tsv'
```

```
python train.py --train_path='./data/CWI 2018 Training Set/english/All_Train.tsv' --dev_path='./data/CWI 2018 Training Set/english/All_Dev.tsv'
```

```
python train.py --train_path='./data/english/All_Train.tsv' --dev_path='./data/english/All_Dev.tsv'
```

###Evaluate

```
python evaluate.py --ckptdir=<dirname> --dataset=<test file>
```

```
ex: python evaluate.py --ckptdir=1548796604 --dataset=All_Test
```


Evaluate the latest checkpoint
```
python evaluate.py --dataset='./data/english/News_Test.tsv' --ckptdir=`ls -lc runs | tail -1| cut -d ' ' -f13` 
```

```
python evaluate.py --dataset='./data/english/WikiNews_Test.tsv' --ckptdir=`ls -lc runs | tail -1| cut -d ' ' -f13` 
```

```
python evaluate.py --dataset='./data/english/Wikipedia_Test.tsv' --ckptdir=`ls -lc runs | tail -1| cut -d ' ' -f13` 
```

```
python evaluate.py --dataset='./data/english/All_Test.tsv' --ckptdir=`ls -lc runs | tail -1| cut -d ' ' -f13` 
```

### References

### Sources


---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)