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




mat_data_01 = scipy.io.loadmat(r'D:\data\data_WB_O_01.mat')
mat_data_02 = scipy.io.loadmat(r'D:\data\data_WB_PO_01.mat')
mat_data_03 = scipy.io.loadmat(r'D:\data\data_WB_UP_01.mat')
mat_data_04 = scipy.io.loadmat(r'D:\data\data_WB_X_01.mat')

txt_data_01 = r'D:\data\O-wendangming.txt'
txt_data_02 = r'D:\data\PO-wendangming.txt'
txt_data_03 = r'D:\data\UP-wendangming.txt'
txt_data_04 = r'D:\data\X-wendangming.txt'



def get_csi_feature(csi_data, filename_time):
    print('data pre_processing')
    time_stamp = get_timestamp(filename_time)
    print(1)
    csi_value = get_csi_value(csi_data)  # 求输入数据的模值 #csi_value(50~~~,270)
    print('csi_value done')
    print('csi_value', csi_value.reshape)  # (124497, 90)
    csi_value_process = data_butter_filter(csi_value)  # input：csi_value：50~~~*270  # output：csi_butter：270*50~~~
    print('csi_value_process', csi_value_process.shape)  # (90, 124497)
    print('buffer process done')
    csi_sample_array = time_stamp_cut(csi_value_process, time_stamp)  # input：270*50~~~ output： (51, 800, 270)
    print('csi_sample_array', csi_sample_array.shape)  # (51, 800, 270)

    csi_pca_temp = ()
    for i in range(csi_sample_array.shape[0]):
        csi_pca_temp = np.append(csi_pca_temp, signal_pca(csi_sample_array[i, :, :]))
    csi_pca = csi_pca_temp.reshape(csi_sample_array.shape[0], 800, 5)
    print('pca done')
    # s = csi_pca[:, :, 1:2]
    # # q = s.shape[2]
    # plt.plot(s[0,:])
    # plt.show()
    feature_data = feature_selection(csi_pca[:, :, 1:5])  # 取第二个主成分
    print('feature_data', feature_data.shape)
    print('feature_selection done')
    return feature_data



feature_data_01 = get_csi_feature(mat_data_01, txt_data_01)
feature_data_02 = get_csi_feature(mat_data_02, txt_data_02)
feature_data_03 = get_csi_feature(mat_data_03, txt_data_03)
feature_data_04 = get_csi_feature(mat_data_04, txt_data_04)


np.save('feature_data_011', feature_data_01)
np.save('feature_data_022' , feature_data_02)
np.save('feature_data_033', feature_data_03)
np.save('feature_data_044' , feature_data_04)
# plt.plot(csi_sample_array[0,:])
# plt.show()




#
#
# feature_data_01 = get_csi_feature(mat_data_01, txt_data_01)
# #
# # np.save('feature_data_011', feature_data_01)
#
#
# plt.plot(abs(feature_data_01))
# plt.show()

