import os, sys
from DATA.lzq_read_data_time_poi import lzq_load_data, prepare_data, intervals_to
import numpy as np
from DST_network.utils import STResNetHelper
from keras import backend as K

K.set_image_data_format('channels_first')

# for reproducibility
np.random.seed(1234)

# paths
# all_data_path = "/content/drive/My Drive/CSE-8673-ML-Data/"
all_data_path = "/home/clinamen/school/ML2020/project/forecast/"
model_save_path = all_data_path

# Hyper-parameters
H, W, channel = 21, 12, 2  # grid size and channels

T = 24 * 1  # number of time intervals in one day
T_closeness, T_period, T_trend = 1, T, T * 7

len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of period dependent sequence
len_trend = 4  # length of trend dependent sequence

# last 7 days for testing data
days_test = 14
len_test = T * days_test


# loading data
all_data = np.load(f"{all_data_path}data_W{W}xH{H}.npy")
poi_data = np.load(os.path.dirname(os.path.realpath(__file__)) + '/DATA/dataBikeNYC/poi_data.npy')

# https://en.wikipedia.org/wiki/COVID-19_pandemic_lockdown_in_Italy
lockdown_start, lockdown_end = '2020-03-09', '2020-05-18'

pre_lockdown_data = all_data[:intervals_to(lockdown_start)]
lockdown_data = all_data[intervals_to(lockdown_start):intervals_to(lockdown_end)]
post_lockdown_data = all_data[intervals_to(lockdown_end):]

# Evaluate STResNet on pre-lockdown data
X_train, T_train, P_train, Y_train, X_test, T_test, P_test, Y_test, MM = prepare_data(pre_lockdown_data, poi_data, len_test,
                                                                                      len_closeness, len_period,
                                                                                      len_trend, T_closeness,
                                                                                      T_period, T_trend)

helper = STResNetHelper('DSTNet', channel, H, W, len_test,
                        len_closeness, len_period,
                        len_trend, T_closeness,
                        T_period, T_trend, model_save_path)

model = helper.build_model()
helper.train(model, X_train, Y_train)
helper.evaluate(model, X_test, Y_test)
