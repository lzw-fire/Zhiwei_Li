import os
from shutil import copyfile

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from HI_data_prepare import get_timestamp

from HI_data_prepare import get_csi_value
from HI_data_prepare import data_butter_filter
from HI_data_prepare import time_stamp_cut
from HI_feature_selection import feature_selection
from HI_data_prepare import signal_pca




O_data_01 = scipy.io.loadmat(r'D:\data\SVM_data/O-WB-01-delay.mat')
# O_data_02 = scipy.io.loadmat(r'D:\data\O-WB-02.mat')
# O_data_03 = scipy.io.loadmat(r'D:\data\O-WB-03.mat')
# O_data_04 = scipy.io.loadmat(r'D:\data\O-WB-04.mat')
# O_data_05 = scipy.io.loadmat(r'D:\data\O-WB-05.mat')

PO_data_01 = scipy.io.loadmat(r'D:\data\SVM_data/PO-WB-01-delay.mat')
# PO_data_02 = scipy.io.loadmat(r'D:\data\PO-WB-02.mat')
# PO_data_03 = scipy.io.loadmat(r'D:\data\PO-WB-03.mat')
# PO_data_04 = scipy.io.loadmat(r'D:\data\PO-WB-04.mat')
# PO_data_05 = scipy.io.loadmat(r'D:\data\PO-WB-05.mat')

UP_data_01 = scipy.io.loadmat(r'D:\data\SVM_data/UP-WB-01-delay.mat')
# UP_data_02 = scipy.io.loadmat(r'D:\data\UP-WB-02.mat')
# UP_data_03 = scipy.io.loadmat(r'D:\data\UP-WB-03.mat')
# UP_data_04 = scipy.io.loadmat(r'D:\data\UP-WB-04.mat')
# UP_data_05 = scipy.io.loadmat(r'D:\data\UP-WB-05.mat')

X_data_01 = scipy.io.loadmat(r'D:\data\SVM_data/X-WB-01-delay.mat')
# X_data_02 = scipy.io.loadmat(r'D:\data\X-WB-02.mat')
# X_data_03 = scipy.io.loadmat(r'D:\data\X-WB-03.mat')
# X_data_04 = scipy.io.loadmat(r'D:\data\X-WB-04.mat')
# X_data_05 = scipy.io.loadmat(r'D:\data\X-WB-05.mat')



O_txt_data_01 = r'D:\data\O-WB-01.txt'
# O_txt_data_02 = r'D:\data\O-WB-02.txt'
# O_txt_data_03 = r'D:\data\O-WB-03.txt'
# O_txt_data_04 = r'D:\data\O-WB-04.txt'
# O_txt_data_05 = r'D:\data\O-WB-05.txt'

PO_txt_data_01 = r'D:\data\PO-WB-01.txt'
# PO_txt_data_02 = r'D:\data\PO-WB-02.txt'
# PO_txt_data_03 = r'D:\data\PO-WB-03.txt'
# PO_txt_data_04 = r'D:\data\PO-WB-04.txt'
# PO_txt_data_05 = r'D:\data\PO-WB-05.txt'

UP_txt_data_01 = r'D:\data\UP-WB-01.txt'
# UP_txt_data_02 = r'D:\data\UP-WB-02.txt'
# UP_txt_data_03 = r'D:\data\UP-WB-03.txt'
# UP_txt_data_04 = r'D:\data\UP-WB-04.txt'
# UP_txt_data_05 = r'D:\data\UP-WB-05.txt'

X_txt_data_01 = r'D:\data\X-WB-01.txt'
# X_txt_data_02 = r'D:\data\X-WB-02.txt'
# X_txt_data_03 = r'D:\data\X-WB-03.txt'
# X_txt_data_04 = r'D:\data\X-WB-04.txt'
# X_txt_data_05 = r'D:\data\X-WB-05.txt'


def get_csi_feature(csi_data, filename_time):
    print('data pre_processing')
    time_stamp = get_timestamp(filename_time)
    # print(1)
    csi_value = get_csi_value(csi_data)  # 求输入数据的模值 #input(270,50---), output csi_value(50---,270)
    print('csi_value done')
    print('csi_value', csi_value.shape)
    # csi_value_process = data_butter_filter(csi_value)  # input：csi_value：50~~~*270  # output：csi_butter：270*50~~~
    # print('csi_value_process', csi_value_process.shape)
    # print('buffer process done')
    csi_sample_array = time_stamp_cut(csi_value, time_stamp)  # input：270*50~~~ output： (51, 800, 270)
    print('time_stamp_cut done')
    # print('csi_sample_array', csi_sample_array.shape)  # (51, 800, 270)

    # csi_pca_temp = ()
    # for i in range(csi_sample_array.shape[0]):
    #     csi_pca_temp = np.append(csi_pca_temp, signal_pca(csi_sample_array[i, :, :]))
    # csi_pca = csi_pca_temp.reshape(csi_sample_array.shape[0], 800, 250)
    # print('pca done')
    # feature_data = csi_sample_array[:,25:775,0:90]
    print('feature_data', csi_sample_array.shape)

    return csi_sample_array



feature_data_01 = get_csi_feature(O_data_01, O_txt_data_01)
feature_data_02 = get_csi_feature(PO_data_01, PO_txt_data_01)
feature_data_03 = get_csi_feature(UP_data_01, UP_txt_data_01)
feature_data_04 = get_csi_feature(X_data_01, X_txt_data_01)


sample_num = 51

feature_1 = feature_data_01[0:sample_num, :, :]
feature_2 = feature_data_02[0:sample_num, :, :]
feature_3 = feature_data_03[0:sample_num, :, :]
feature_4 = feature_data_04[0:sample_num, :, :]
# feature_5 = feature_5[0:sample_num, :]

label_1 = np.ones(len(feature_1)) * 0
label_2 = np.ones(len(feature_2)) * 1
label_3 = np.ones(len(feature_3)) * 2
label_4 = np.ones(len(feature_4)) * 3
# label_5 = np.ones(len(feature_5)) * 5


import random

index = np.arange(sample_num)
random.shuffle(index)

feature_1_index = feature_1[index]
feature_2_index = feature_2[index]
feature_3_index = feature_3[index]
feature_4_index = feature_4[index]
# feature_5_index = feature_5[index]


label_1_index = label_1[index]
label_2_index = label_2[index]
label_3_index = label_3[index]
label_4_index = label_4[index]
# label_5_index = label_5[index]


train_num = 41

X = np.vstack((feature_1_index, feature_2_index, feature_3_index,feature_4_index))  #在列上合并
Y = np.hstack((label_1_index, label_2_index, label_3_index,label_4_index))  #在行上合并

# train_X = np.vstack((feature_1_index[0:train_num, :], feature_2_index[0:train_num, :], feature_3_index[0:train_num, :],
#                      feature_4_index[0:train_num, :]))  #在列上合并
# train_Y = np.hstack((label_1_index[0:train_num], label_2_index[0:train_num], label_3_index[0:train_num],
#                      label_4_index[0:train_num]))  #在行上合并
# test_X = np.vstack((feature_1_index[train_num:sample_num, :], feature_2_index[train_num:sample_num, :],
#                     feature_3_index[train_num:sample_num, :], feature_4_index[train_num:sample_num, :]))
# test_Y = np.hstack((label_1_index[train_num:sample_num], label_2_index[train_num:sample_num],
#                     label_3_index[train_num:sample_num], label_4_index[train_num:sample_num]))


# np.save('train_X', train_X)
# np.save('train_Y' , train_Y)
# np.save('test_X' , test_X)
# np.save('test_Y' , test_Y)

np.save('O-01-delay-X', X)
np.save('O-01-delay-Y' , Y)
# plt.plot(csi_sample_array[0,:])
# plt.show()