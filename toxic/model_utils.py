from keras.layers import LSTM, Input, Embedding, Dense,\
                         Bidirectional, BatchNormalization, Activation,\
                         Dropout, GRU
from keras.models import Model
from keras.optimizers import Nadam


def get_model(sequence_len, dense_shape, word_embeddings_matrix, lstm_dim,
              RNN=LSTM):
    text_input = Input(shape=(sequence_len,))
    embedding_layer = Embedding(word_embeddings_matrix.shape[0],
                                word_embeddings_matrix.shape[1],
                                trainable=False,
                                weights=[word_embeddings_matrix])
    x = embedding_layer(text_input)
    x = Bidirectional(RNN(lstm_dim, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    x = Dense(dense_shape)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    model_prediction = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=[text_input], outputs=[model_prediction])
    model.compile(optimizer=Nadam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model
