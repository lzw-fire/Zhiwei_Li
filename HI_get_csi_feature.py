
import os
from shutil import copyfile

import numpy as np
import scipy.io
from HI_data_prepare import get_timestamp

from HI_data_prepare import get_csi_value
from HI_data_prepare import data_butter_filter
from HI_data_prepare import time_stamp_cut
from HI_feature_selection import feature_selection
from HI_data_prepare import signal_dif


mat_data_01 = scipy.io.loadmat(r'D:\data\SVM_data\O-WB-01-angle.mat')
mat_data_02 = scipy.io.loadmat(r'D:\data\SVM_data\PO-WB-01-angle.mat')
mat_data_03 = scipy.io.loadmat(r'D:\data\SVM_data\UP-WB-01-angle.mat')
mat_data_04 = scipy.io.loadmat(r'D:\data\SVM_data\X-WB-01-angle.mat')




txt_data_01 = r'D:\data\O-WB-01.txt'
txt_data_02 = r'D:\data\PO-WB-01.txt'
txt_data_03 = r'D:\data\UP-WB-01.txt'
txt_data_04 = r'D:\data\X-WB-01.txt'



def get_csi_feature(csi_data, filename_time):
    print(0)
    time_stamp = get_timestamp(filename_time)
    print(1)
    csi_value = get_csi_value(csi_data)  # 求输入数据的模值 #csi_value(50~~~,270)
    # csi_ph_dif = get_csi_data_ph_dif(csi_value)  #50```*2
    # print(2)
    # print('csi_value',csi_value.shape)#(124497, 90)
    # csi_value_process = data_butter_filter(csi_ph_dif) # input：csi_value：50~~~*270  # output：csi_butter：270*50~~~
    # print('csi_value_process',csi_value_process.shape)#(90, 124497)
    # print(3)
    csi_dif = signal_dif(csi_value)

    csi_sample_array = time_stamp_cut(csi_dif, time_stamp)   # input：270*50~~~ output： (51, 800, 270)
    print('csi_sample_array',csi_sample_array.shape)#(51, 800, 270)

    # csi_pi_dif = ()
    # for i in range(csi_sample_array.shape[0]):
    #     csi_pca_temp = np.append(csi_pi_dif, get_csi_data_ph_dif(csi_sample_array[i, :, :]))
    # csi_p = csi_pi_dif.reshape(csi_sample_array.shape[0], 800, 2)
    # csi_pca = csi_p[:,:,1]
    # print('pca done')

    feature_data = feature_selection(csi_sample_array[:,:,1])
    print(4)

    return feature_data

# def get_csi_feature(csi_data, filename_time):
#     csi_value_process = mat_data_01
#     print(0)
#     time_stamp = get_timestamp(filename_time)
#     print(1)
#     csi_value = get_csi_value(csi_data)
#     print(2)
#     print('csi_value',csi_value.shape)
#     csi_value_process = data_butter_filter(csi_value)
#     print(3)
#     print('csi_value_process',csi_value_process.shape)
#     # csi_pca = signal_pca(csi_value_process.T)
#     # print(4)
#     csi_sample_array = time_stamp_cut(csi_value_process, time_stamp)
#     print('csi_sample_array',csi_sample_array.shape)
#     print(5)
#     feature_data = feature_selection(csi_sample_array)
#     print('feature_data',feature_data.shape)
#
#     return feature_data


feature_data_01 = get_csi_feature(mat_data_01, txt_data_01)
feature_data_02 = get_csi_feature(mat_data_02, txt_data_02)
feature_data_03 = get_csi_feature(mat_data_03, txt_data_03)
feature_data_04 = get_csi_feature(mat_data_04, txt_data_04)




np.save('D:/data/SVM_data/O-WB-01-angle-svm', feature_data_01)
np.save('D:/data/SVM_data/PO-WB-01-angle-svm', feature_data_02)
np.save('D:/data/SVM_data/UP-WB-01-angle-svm', feature_data_03)
np.save('D:/data/SVM_data/X-WB-01-angle-svm', feature_data_04)

