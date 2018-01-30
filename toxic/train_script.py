from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import argparse
from toxic.model_utils import get_model
import numpy as np
import pickle


def main():
    parser = argparse.ArgumentParser(
        description="LSTM for identifying and \
                     classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("val_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("--result-path", default="./results/")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=200)
    parser.add_argument("--recurrent-units", type=int, default=60)
    parser.add_argument("--dense-size", type=int, default=30)

    args = parser.parse_args()

    train_archive = np.load(args.train_file_path)
    val_archive = np.load(args.val_file_path)
    embedding_weights = np.load(args.embedding_path)['weights']

    train_text, train_targets = train_archive['text'][0:1000], train_archive['targets'][0:1000]
    val_text, val_targets = val_archive['text'], val_archive['targets']

    model = get_model(sequence_len=args.sentences_length,
                      dense_shape=args.dense_size,
                      word_embeddings_matrix=embedding_weights,
                      lstm_dim=args.recurrent_units)

    checkpoint = ModelCheckpoint(filepath=args.result_path + "weights/",
                                 save_best_only=True,
                                 save_weights_only=True)
    print("Training Model...")
    history = model.fit(train_text, train_targets, epochs=25,
              batch_size=args.batch_size,
              validation_data=(val_text, val_targets),
              callbacks=[checkpoint, ReduceLROnPlateau()])

    print("Saving History")
    pickle.dump(history, open(args.result_path + "history/" +
                              "model_history.pkl",'wb'))

if __name__ == "__main__":
    main()
