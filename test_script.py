import pandas as pd
from toxic.model_utils import get_model_attention, get_model, mean_auc
import numpy as np
from keras.layers import GRU, LSTM
import pickle, argparse
from toxic.metrics import EpochAUC

parser = argparse.ArgumentParser(
    description="Script for testing toxic classifier models")

parser.add_argument("model_weights_path")
parser.add_argument("val_file_path")
parser.add_argument("word_index_path")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--attention", default='No')
parser.add_argument("--sentences-length", type=int, default=200)               
parser.add_argument("--recurrent-units", type=int, default=30)                 
parser.add_argument("--dense-size", type=int, default=20)                      
parser.add_argument("--rnn", default='LSTM')                                   
parser.add_argument("--dropout", type=float, default=0.2)                      
parser.add_argument("--train_embeddings", default='True')                      
parser.add_argument("--lr", type=int, default=0.001)                           
parser.add_argument("--word_index_path", default="None")                       
parser.add_argument("--embeddings_dim", default=200, type=int)                           
parser.add_argument("--pairwise_penalty", type=bool, default=False)

args = parser.parse_args()
rnn_dict = {"LSTM": LSTM, "GRU": GRU}
word_index = pickle.load(open(args.word_index_path, 'rb'))

model = get_model_attention(sequence_len=args.sentences_length,
                      dense_shape=args.dense_size,
                      lstm_dim=args.recurrent_units,
                      RNN=rnn_dict[args.rnn],
                      word_index=word_index,
                      embeddings_dim=args.embeddings_dim,
                      dropout_rate=args.dropout,
                      word_embeddings_matrix=None)

model.load_weights(args.model_weights_path)


val_data = np.load(args.val_file_path)
val_text = val_data['text']

predictions = model.predict(val_text, batch_size=args.batch_size, verbose=1)
print("Validation mean AUC:", mean_auc(val_data['targets'], predictions))

