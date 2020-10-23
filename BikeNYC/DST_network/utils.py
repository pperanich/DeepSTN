import numpy as np
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from DST_network.STResNet import stresnet
import DST_network.metrics as metrics


class BaseHelper:
    def __init__(self, model_name, channel, H, W, len_test,
                 len_closeness, len_period,
                 len_trend, T_closeness,
                 T_period, T_trend, model_save_path, lr=0.0002):
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.lr = lr
        self.W = W
        self.H = H
        self.channel = channel
        self.len_test = len_test
        self.len_period = len_period
        self.len_closeness = len_closeness
        self.len_trend = len_trend
        self.T_period = T_period
        self.T_closeness = T_closeness
        self.T_trend = T_trend

    def _get_model_filename(self):
        return self.model_save_path + '/' + self.model_name + '.hdf5'

    def _get_model_checkpoint(self):
        return ModelCheckpoint(
            filepath=self._get_model_filename(),
            monitor='val_rmse',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1)

    def load(self, model):
        model.load_weights(self._get_model_filename())


class STResNetHelper(BaseHelper):
    def build_model(self, external_dim=False, R_N=4, CF=64):
        c_conf = (self.len_closeness, self.channel, self.H, self.W) if self.len_closeness > 0 else None
        p_conf = (self.len_period, self.channel, self.H, self.W) if self.len_period > 0 else None
        t_conf = (self.len_trend, self.channel, self.H, self.W) if self.len_trend > 0 else None

        model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                         external_dim=external_dim, nb_residual_unit=R_N, CF=CF)

        adam = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse, metrics.mae])
        # model.summary()
        return model

    def train(self, model, X_train, Y_train, epochs=10, batch_size=32):
        model.fit(X_train, Y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  callbacks=[self._get_model_checkpoint()],
                  verbose=1)

    def evaluate(self, model, X_test, Y_test):
        score = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
        print('(loss, rmse, mse):', end=' ')
        np.set_printoptions(precision=6, suppress=True)
        print(np.array(score))
