import keras.backend as K

def keras_auc(y_true, y_pred):
    auc_list = K.tf.metrics.auc(predictions=y_true, labels=y_pred)
    return K.tf.constant(auc_list[1])
