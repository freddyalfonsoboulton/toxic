from keras.callbacks import Callback
from .model_utils import mean_auc

class EpochAUC(Callback):
    def __init__(self, validation_data):
        super(Callback, self).__init__()
        self.validation_data, self.validation_y = validation_data


    def on_epoch_end(self, epoch, logs={}):
        validation_preds = self.model.predict(self.validation_data,
                                              batch_size=512, verbose=0)
        auc = mean_auc(self.validation_y, validation_preds)
        print("AUC after epoch is {0:.5f}".format(auc))
