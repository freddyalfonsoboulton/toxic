from keras.callbacks import Callback
from .model_utils import mean_auc

class EpochAUC(Callback):
    def on_epoch_end(self):
        valid_x, valid_x, _ = self.model.validation_data
        validation_preds = self.model.predict(valid_x,
                                              batch_size=512, verbose=0)
        auc = mean_auc(valid_x, validation_preds)
        print("AUC after epoch is {0:.5f}".format(auc))
