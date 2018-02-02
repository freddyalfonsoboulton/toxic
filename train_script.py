from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,\
                            EarlyStopping
from keras.layers import GRU, LSTM
import argparse
from toxic.model_utils import get_model, get_model_attention
import numpy as np
import pickle


def main():
    parser = argparse.ArgumentParser(
        description="LSTM for identifying and \
                     classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("val_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("--attention", default='No')
    parser.add_argument("--result-path", default="./results/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sentences-length", type=int, default=200)
    parser.add_argument("--recurrent-units", type=int, default=30)
    parser.add_argument("--dense-size", type=int, default=20)
    parser.add_argument("--rnn", default='LSTM')
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    rnn_dict = {"LSTM": LSTM, "GRU": GRU}
    attention_dict = {'No': get_model, 'Yes': get_model_attention}

    train_archive = np.load(args.train_file_path)
    val_archive = np.load(args.val_file_path)
    embedding_weights = np.load(args.embedding_path)['weights']

    train_text, train_targets = train_archive['text'],\
                                train_archive['targets']
    val_text, val_targets = val_archive['text'],\
                            val_archive['targets']

    func = attention_dict[args.attention]

    model = func(sequence_len=args.sentences_length,
                 dense_shape=args.dense_size,
                 word_embeddings_matrix=embedding_weights,
                 lstm_dim=args.recurrent_units,
                 RNN=rnn_dict[args.rnn],
                 dropout_rate=args.dropout)

    checkpoint = ModelCheckpoint(filepath=args.result_path + "weights/" +
                                 "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 save_best_only=True,
                                 save_weights_only=True,
                                 verbose=1)
    es = EarlyStopping(patience=2, verbose=1)
    print("Training Model...")
    history = model.fit(train_text, train_targets, epochs=10,
                        batch_size=args.batch_size,
                        validation_data=(val_text, val_targets),
                        callbacks=[checkpoint, ReduceLROnPlateau(), es])

    print("Saving History")
    pickle.dump(history.history, open(args.result_path + "history/" +
                                      "model_history.pkl", 'wb'))


if __name__ == "__main__":
    main()
