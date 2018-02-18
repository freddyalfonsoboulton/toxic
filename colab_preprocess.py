from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import get_file
import argparse
import pandas as pd
import numpy as np
import os

parser = argparse.ArgumentParser(description="Preprocessing kaggle-toxic data")
parser.add_argument("train_data_url")
parser.add_argument("test_data_url")
args = parser.parse_args()


#make file structure
if not os.path.exists('toxic/data'):
    os.mkdir('toxic/data')
if not os.path.exists('toxic/data/raw'):
    os.mkdir('toxic/data/raw')
if not os.path.exists('toxic/data/processed'):
    os.mkdir('toxic/data/processed')

get_file('train.csv.zip', origin=args.train_data_url, 
    cache_dir='toxic/data/raw', cache_subdir='./')
get_file('test.csv.zip', origin=args.test_data_url, 
    cache_dir='toxic/data/raw', cache_subdir='./')

MAX_SEQUENCE_LEN = 200

train = pd.read_csv('toxic/data/raw/train.csv.zip')
test = pd.read_csv('toxic/data/raw/test.csv.zip')

train_corpus = list(train.comment_text.values)
test_corpus = list(test.comment_text.values)
corpus = train_corpus + test_corpus

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
train_sequences = tokenizer.texts_to_sequences(train_corpus)
test_sequences = tokenizer.texts_to_sequences(test_corpus)

np.random.seed(2)
train_indices = np.random.choice(train.id,int(0.9*train.shape[0]),replace=False)

train_seq = np.array(train_sequences)[train.id.isin(train_indices)].tolist()
val_seq = np.array(train_sequences)[~train.id.isin(train_indices)].tolist()

headers = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
train_targets = train.loc[train.id.isin(train_indices),headers].values
val_targets = train.loc[~train.id.isin(train_indices),headers].values

train_seq_padded = pad_sequences(train_seq,maxlen=MAX_SEQUENCE_LEN)
val_seq_padded = pad_sequences(val_seq,maxlen=MAX_SEQUENCE_LEN)

test_sequences_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LEN)

np.savez_compressed("toxic/data/processed/train_data.npz",text = train_seq_padded, targets=train_targets)
np.savez_compressed("toxic/data/processed/val_data.npz", text= val_seq_padded, targets = val_targets)
np.savez_compressed("toxic/data/processed/test_data.npz", text=test_sequences_padded)

import pickle

pickle.dump(tokenizer.word_index, open("toxic/data/processed/word_index_dictionary.pkl",'wb'))

