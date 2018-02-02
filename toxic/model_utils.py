from keras.layers import LSTM, Input, Embedding, Dense,\
                         Bidirectional, BatchNormalization, Activation,\
                         Dropout, GRU
from keras.models import Model
from keras.optimizers import Nadam


def get_model(sequence_len, dense_shape, word_embeddings_matrix, lstm_dim,
              dropout_rate, RNN=LSTM):
    """Vanilla architecture for LSTM based on word embeddings."""
    text_input = Input(shape=(sequence_len,))
    embedding_layer = Embedding(word_embeddings_matrix.shape[0],
                                word_embeddings_matrix.shape[1],
                                trainable=False,
                                weights=[word_embeddings_matrix])
    x = embedding_layer(text_input)
    x = Bidirectional(RNN(lstm_dim, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_shape)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    model_prediction = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=[text_input], outputs=[model_prediction])
    model.compile(optimizer=Nadam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def get_model_attention(sequence_len, dense_shape, word_embeddings_matrix,
                        lstm_dim, dropout_rate, RNN=LSTM):
    """Model that attends to first RNN's sequence."""
    text_input = Input(shape=(sequence_len,))
    embedding_layer = Embedding(word_embeddings_matrix.shape[0],
                                word_embeddings_matrix.shape[1],
                                trainable=False,
                                weights=[word_embeddings_matrix])
    x = embedding_layer(text_input)
    x = Bidirectional(RNN(lstm_dim, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(RNN(lstm_dim, return_sequences=False))(x)
    x = Dense(dense_shape, activation='relu')(x)
    model_prediction = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=[text_input], outputs=[model_prediction])
    model.compile(optimizer=Nadam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model
