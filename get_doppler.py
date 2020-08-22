import numpy as np
import scipy.io
import math
from HI_data_prepare import get_timestamp
from HI_data_prepare import time_stamp_cut
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.decomposition import PCA
import tftb


rx = 3
tx = 3
subcarrier_num = 30

O_data_01 = scipy.io.loadmat(r'D:\data\X-WB-01.mat')
O_txt_data_01 = r'D:\data\X-WB-01.txt'

csi_data_name = list(O_data_01.keys())[3]
csi_data = O_data_01[csi_data_name].T

# time_stamp = get_timestamp(O_txt_data_01)
# csi_sample_array = time_stamp_cut(csi_data, time_stamp)
# sample_num = csi_sample_array.shape[0]
# sample_len = csi_sample_array.shape[1]
csi = csi_data[0:800,0:90]
# csi_shift = np.angle(csi)
# plt.plot(abs(csi[:,1]))
# plt.show()


 # Select Antenna Pair[WiDance]
csi_mean = np.mean(abs(csi),0)
csi_var = np.sqrt(np.var(abs(csi),0))
csi_mean_var_ratio = csi_mean/csi_var
idx = np.argmax(np.mean(csi_mean_var_ratio.reshape(subcarrier_num, rx, order='F'),0)) + 1
csi_ref = np.tile(csi[:,(idx-1)*30:idx*30], (1, rx))


# Amp Adjust[IndoTrack]
csi_adj = np.zeros(np.size(csi), dtype=complex).reshape(csi.shape[0],csi.shape[1])
csi_ref_adj = np.zeros(np.size(csi_ref), dtype=complex).reshape(csi_ref.shape[0],csi_ref.shape[1])
alpha_sum = 0
for j in range(subcarrier_num*rx):
    amp = abs(csi[:,j])
    alpha = np.min(amp)
    alpha_sum += alpha
    csi_adj[:,j] = abs(abs(csi[:,j]) - alpha) * np.exp(1j * np.angle(csi[:,j]))

beta = 1000 * alpha_sum / (30 * rx)
for j in range(subcarrier_num*rx):
    csi_ref_adj[:,j] = (abs(csi_ref[:,j]) + beta) * np.exp(1j * np.angle(csi_ref[:,j]))

# Conj Mult 共轭相乘
conj_mult = csi_adj * csi_ref_adj.conjugate()
conj_mult = np.append(conj_mult[:,0:30*(idx - 1)], conj_mult[:,30*idx+0:90], axis=1)

# Filter Out Static Component & High Frequency Component
samp_rate = 1000
half_rate = samp_rate / 2
uppe_orde = 6
uppe_stop = 60
lowe_orde = 3
lowe_stop = 2

lu, ld = signal.butter(uppe_orde, uppe_stop / half_rate, 'low', analog=False) #第一个为阶数，Wn是归一化频率，具体计算方法是（2*截止频率)/采样频率
hu, hd = signal.butter(lowe_orde, lowe_stop / half_rate, 'high', analog=False)
for j in range(conj_mult.shape[1]):
    conj_mult[:,j] = signal.filtfilt(lu, ld, conj_mult[:,j])
    conj_mult[:,j] = signal.filtfilt(hu, hd, conj_mult[:,j])

# PCA analysis
# pca_coef = pca(conj_mult);
# conj_mult_pca = conj_mult * pca_coef(:,1);
pca = PCA(n_components=1)
conj_mult_pca = pca.fit_transform(abs(conj_mult))

# STFT
time_instance = conj_mult_pca.shape[0]
window_size = round(samp_rate/4+1)
if (not (window_size % 2)):
    window_size = window_size + 1

f, t, Zxx = signal.stft(conj_mult_pca.T, fs=samp_rate, window='hann', nperseg=window_size)
Zxx = Zxx.reshape(Zxx.shape[1], Zxx.shape[2])
plt.plot(abs(Zxx))
plt.show()
cm_light = plt.cm.get_cmap('rainbow')
ct = plt.pcolormesh(t,np.abs(Zxx), cmap=cm_light ,edgecolors='face')
plt.colorbar(ct)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
plt.show()
# plt.plot(Zxx[0,:,:])
# plt.show()

print(csi.shape())
