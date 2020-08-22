# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:19:51 2018

@author: XueDing
"""
import numpy as np
import pandas as pd
import math
from HI_data_prepare import signal_pca
import scipy.signal as signal


# sample=np.array([[1,2,3,4,1,2,0],[2,3,4,5,6,7,8],[2,3,5,6,8,9,0],[7,6,5,4,3,3,3]])

def feature_selection(sample):

    # input: 51*800
    num = sample.shape[1]
    # print(num)
    '''时域特征'''
    # energy能量
    csi_energy = np.sum(abs(sample[:, 0:num]) ** 2, axis=1) # 51
    # print(csi_energy)
    # print(csi_energy.shape)
    # print(type(csi_energy))

    # excess delay附加时延

    # RMS_Delay均方根时延

    # 均值mean
    csi_mean = np.mean(abs(sample[:, 0:num]), axis=1) # 51
    # print(csi_mean)

    # 最大值max
    csi_max = np.max(abs(sample[:, 0:num]), axis=1) # 51
    # print(csi_max)

    #最小值min
    csi_min = np.min(abs(sample[:, 0:num]), axis=1)  # 51

    # std 标准方差
    csi_std = np.std(abs(sample[:, 0:num]), axis=1) # 51
    # print(csi_std)

    # peak 峰差
    csi_peak = csi_max - csi_min

    # 均方根 rms(有效值）
    ipf = np.power(abs(sample), 2)
    imean = np.mean(ipf[:, 0:num], axis=1)
    csi_rms = np.sqrt(imean)

    # 峰值因子 cresfactor
    csi_crestfactor = csi_peak / csi_rms

    # 波形因子 shapefactor
    csi_shapefactor = csi_rms / csi_mean

    # 脉冲因子 impulsefactor
    csi_impulsefactor = csi_peak / csi_mean

    # 裕度因子 marginfactor
    s1 = np.mean(np.sqrt(abs(sample))[:, 0:num], axis=1)
    s2 = np.power(s1, 2)
    csi_marginfactor = csi_peak / s2



    # kurtosis峰度
    csi_kur = np.zeros((sample.shape[0]))   # 51
    for single_sample in range(sample.shape[0]):  #sample.shape[0]=0-51
        s1 = pd.Series(sample[single_sample, 0:num])
        kur = s1.kurt()
        csi_kur[single_sample] = kur  # 51
        # print(csi_kur)

    # skewness偏度
    csi_skew = np.zeros((sample.shape[0]))

    for single_sample in range(sample.shape[0]):
        s2 = pd.Series(sample[single_sample, 0:num])
        ske = s2.skew()
        csi_skew[single_sample] = ske  # 51
        # print(csi_skew)

    # 相对于信号峰值衰减10db的信号帧数
    threshold_db = -10
    db_num = 0
    # csi_db_num = []
    csi_db_num=np.zeros((sample.shape[0])) # 51
    temp_abs = abs(sample[:, 0:num])  #  51*800
    temp_thread = pow(10, threshold_db / 10) * np.max(temp_abs, axis=1)  # 51

    for j in range(sample.shape[0]):
        g = 0
        for k in range(sample.shape[1]):
            if temp_abs[j, k] > temp_thread[j]:
                g = g + 1
        csi_db_num[j] = g  #

    # for i in range(dim):
    #     for j in range(sample.shape[0]):
    #         temp_num = temp_abs[j][temp_abs[j, i] > temp_thread[j, i]]
    #         db_num = temp_num.size
    #         csi_db_num = np.append(csi_db_num, db_num)
    #     # print(csi_db_num)


    '''计算熵'''
    # 统计已知数据中的不同数据及其出现次数
    csi_entropyVal = np.zeros((sample.shape[0]))
    for single_sample in range(sample.shape[0]):
        dataArrayLen = len(sample[single_sample])
        diffData = []
        diffDataNum = []
        dataCpy = list(sample[single_sample])
        for i in range(dataArrayLen):
            count = 0
            j = i
            if (dataCpy[j] != '*'):
                temp = dataCpy[i]
                diffData.append(temp)
                while (j < dataArrayLen):
                    if (dataCpy[j] == temp):
                        count = count + 1
                        dataCpy[j] = '*'
                    j = j + 1
                diffDataNum.append(count)

        # 计算已知数据的熵
        diffDataArrayLen = len(diffDataNum)
        for i in range(diffDataArrayLen):
            proptyVal = diffDataNum[i] / dataArrayLen
            csi_entropyVal[single_sample] = csi_entropyVal[single_sample] - proptyVal * math.log2(proptyVal)
    '''频域特征'''


    feature = np.vstack((csi_energy, csi_mean, csi_max, csi_std, csi_kur, csi_skew, csi_db_num)) #, csi_rms #, csi_crestfactor,csi_impulsefactor,csi_marginfactor, csi_shapefactor)) #51*28
    # print(feature)

    return feature.T

# aa=feature_selection(sample)










