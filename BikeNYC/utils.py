import numpy as np
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from DST_network.STResNet import stresnet
import DST_network.metrics as metrics
from DeepSTN_network.DeepSTN_net import DeepSTN


class BaseHelper:
    def __init__(self, model_name, channel, H, W, len_test,
                 len_closeness, len_period,
                 len_trend, T_closeness,
                 T_period, T_trend, model_save_path, lr=0.0002, T=24):
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
        self.T = T

    def _get_model_filename(self):
        return self.model_save_path + '/' + self.model_name + '.hdf5'

    def _get_model_checkpoint(self):
        return ModelCheckpoint(
            filepath=self._get_model_filename(),
            monitor='val_rmse',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            period=1)

    def load(self, model):
        model.load_weights(self._get_model_filename())

    def evaluate(self, model, X_test, Y_test):
        score = model.evaluate(self.to_model_input(X_test), Y_test, batch_size=32, verbose=0)
        print(dict(zip(model.metrics_names, score)))

    def predict(self, model, X):
        return model.predict(self.to_model_input(X))

    def train(self, model, X_train, Y_train, epochs=10, batch_size=32):
        model.fit(self.to_model_input(X_train), Y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  callbacks=[self._get_model_checkpoint()],
                  verbose=1)

    def to_model_input(self, X):
        return X


class DeepSTNHelper(BaseHelper):
    def build_model(self, is_plus=False):
        pre_F = 64
        conv_F = 64
        R_N = 2

        plus = 8
        rate = 1

        is_pt = False
        P_N = 9
        T_F = 7 * 8
        PT_F = 9

        drop = 0.1

        model = DeepSTN(H=self.H, W=self.W, channel=self.channel,
                        c=self.len_closeness, p=self.len_period,
                        pre_F=pre_F, conv_F=conv_F, R_N=R_N,
                        is_plus=is_plus,
                        plus=plus, rate=rate,
                        is_pt=is_pt, P_N=P_N, T_F=T_F, PT_F=PT_F, T=self.T,
                        drop=drop, is_summary=False)

        return model

    def to_model_input(self, X):
        return np.concatenate((X[0], X[1], X[2]), axis=1)


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
