from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import GRU, LSTM
import argparse
from toxic.model_utils import get_model, get_model_attention
import numpy as np
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="RNN for identifying and \
                     classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("weights_file")
    parser.add_argument("submission_path")
    parser.add_argument("--embedding_path", default="None")
    parser.add_argument("--attention", default='No')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sentences-length", type=int, default=200)
    parser.add_argument("--recurrent-units", type=int, default=30)
    parser.add_argument("--dense-size", type=int, default=20)
    parser.add_argument("--rnn", default='LSTM')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train_embeddings", default='True')
    parser.add_argument("--lr", type=int, default=0.001)


    args = parser.parse_args()

    assert os.path.isdir(args.submission_path), "submission_path does not exist"

    rnn_dict = {"LSTM": LSTM, "GRU": GRU}
    attention_dict = {'No': get_model, 'Yes': get_model_attention}

    train_archive = np.load(args.train_file_path)
    test_archive = np.load(args.test_file_path)

    if args.embedding_path == "None":
        embedding_weights = None
    else:
        embedding_weights = np.load(args.embedding_path)['weights']

    train_text, train_targets = train_archive['text'], train_archive['targets']
    test_text = test_archive['text']

    func = attention_dict[args.attention]

    model = func(sequence_len=args.sentences_length,
                 dense_shape=args.dense_size,
                 word_embeddings_matrix=embedding_weights,
                 lstm_dim=args.recurrent_units,
                 RNN=rnn_dict[args.rnn],
                 dropout_rate=args.dropout,
                 trainable_embeddings=bool(args.train_embeddings),
                 lr=args.lr)

    model.load_weights(args.weights_file)

    es = EarlyStopping(patience=2, verbose=1)
    print("Training Model...")
    history = model.fit(train_text, train_targets, epochs=4,
                        batch_size=args.batch_size,
                        callbacks=[ReduceLROnPlateau(), es])

    print("Making Predictions")
    preds = model.predict(test_text, batch_size=512, verbose=1)

    submission_file = pd.read_csv("data/sample_submission.csv")

    names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
             'identity_hate']
    submission_file = pd.concat([submission_file.id,
                                 pd.DataFrame(preds, columns=names)], axis=1)
    submission_file.to_csv(args.submission_path, index=False)

if __name__ == "__main__":
    main()
