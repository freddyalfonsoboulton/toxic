from keras import backend as K
import numpy as np
import itertools

# 6 choose 2
number_combinations = 15
all_pairwise_transform = np.zeros((number_combinations, 6))
for n, (i, ii) in enumerate(itertools.combinations(range(6), 2)):
    all_pairwise_transform[n, [i, ii]] = 0.5, 0.5

tf_pairwise = K.tf.constant(all_pairwise_transform.T, dtype='float32')


def pairwise_deviation(y_true, y_pred):
    "Compute deviation of pairwise probabilities from true 'pairwise' label."
    true_pairwise = K.tf.matmul(y_true, tf_pairwise)
    pred_pairwise = K.tf.matmul(y_pred, tf_pairwise)
    diff = K.tf.subtract(pred_pairwise, true_pairwise)
    diff = K.tf.square(diff)
    return K.expand_dims(K.tf.reduce_sum(diff, axis=-1), axis=-1)


def entropy_plus_pairwise_dev(y_true, y_pred):
    ""
    return K.binary_crossentropy(y_true, y_pred) +\
           pairwise_deviation(y_true, y_pred)
