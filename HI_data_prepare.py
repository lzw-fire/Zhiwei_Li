# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:48:44 2018

@author: XueDing
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io

#===================================================

def A(x, y):
    sum_pow = pow(x, 2) + pow(y, 2)
    square = np.sqrt(sum_pow)
    return square

def get_csi_value(csi_data):  # 用于scipy.io.loadmat  input(270, 50---)
    csi_data_name = list(csi_data.keys())[3]#print(csi_data.keys()):dict_keys(['__header__', '__version__', '__globals__', 'csi_amp'])
    len_csi = csi_data[csi_data_name].shape[1] #(270, 50~~~)
    csi_complex = csi_data[csi_data_name]
    # csi_complex = csi_complex.T  # (50~~~, 90) 可以根据时间戳将每个样本截成或补成700长度 ，再抽样成100
    #csi_value = np.zeros((len_csi, 90))
    #csi_value = A(csi_complex.real, csi_complex.imag)
    # csi_value = abs(csi_complex)

    return csi_complex  # (50~~~, 270)

#================================================
def data_butter_filter(csi_value):
    # input：csi_value：50~~~*270
    # output：sig_ff：270*50~~~

    csi_value = csi_value.T
    csi_value_shape = csi_value.shape[0]

    b, a = signal.butter(5, 0.1, 'low', analog=False) #Wn是归一化频率，具体计算方法是（2*截止频率)/采样频率，2*50/1000=0.1，采样频率是多少？
    # b, a = signal.butter(8, 0.1,'high', analog=False)这个可以把高频噪声提取出来

    sig_f = []
    for i in range(csi_value_shape):
        #print(i)
        sig = csi_value[i, :]#每个（270）子载波分别进行滤波
        #print(sig.shape)#(50~~~,)
        sig_f.append(signal.filtfilt(b, a, sig))
        #print(len(sig_f))#i+1
        sig_ff = np.array(sig_f)
        #print(sig_ff.shape)#(i+1, 50~~~)
    return sig_ff #(270, 50~~~)


def get_time(time_line):
    return time_line.split('****')[0]


def get_timestamp(filename):
    time_stamp = []
    with open(filename) as f:
        line = f.readline()
        while line:
            time = int(get_time(line))
            time_stamp.append(time)
            line = f.readline()
        return time_stamp
        # print(time_stamp)

def time_stamp_cut(csi_value_process, time_stamp):
    # 根据wendangming.txt时间戳截取csi模值，每1000帧截取为一个样本
    # input：csi_value_process：270*50~~~
    # output：sample：51*800*270 (51, 800, 270)

    csi_value_process = csi_value_process.T  #（270*50---）
    # csi_value_process = np.array(csi_value_process)

    csi_sample = []
    csi_sampled = []
    csi_cut_len = 1000
    csi_sample_len = 800  # 如果最后一个动作小于这个长度，就会报错，reshape的时候会有问题
    for j in range(len(time_stamp) - 1):  #j:0-50
        # print(j)
        left_time = time_stamp[j]
        right_time = left_time + csi_cut_len
        csi_cut = csi_value_process[left_time:right_time,:]  # 这一步即使right_time超过了csi_value_process的右侧边界的index也不会报错，只是截取到右侧边界赋值
        csi_sample = csi_cut[0:csi_sample_len,:]#默认截取行
        csi_sampled.extend(csi_sample)

    # print('csi_sampled',len(csi_sampled))
    csi_sample_array = np.array(csi_sampled)
    # print('csi_sample_array',csi_sample_array.shape)
    # sample = csi_sample_array.reshape(len(time_stamp)- 1, csi_sample_len,-1)
    sample = csi_sample_array.reshape(len(time_stamp) - 1, csi_sample_len, 3)
    return sample


def signal_pca(csi_value, components):
    # 对获取的csi模值进行去噪和降维处理
    # input：csi_value：800*270
    # return：csi_value_pca：800*1 去最大的主成分，降为1维

    from sklearn.decomposition import PCA
    csi_value_pca = []
    pca = PCA(n_components=components)
    csi_value_pca = pca.fit_transform(csi_value)
    return csi_value_pca

def signal_dif(csi_vale):
    csi_vale_dif = np.zeros([csi_vale.shape[0], csi_vale.shape[1]-1])
    for i in range(csi_vale.shape[1]-1):
        csi_vale_dif[:,i] = csi_vale[:,i+1] - csi_vale[:,i]
    return csi_vale_dif





