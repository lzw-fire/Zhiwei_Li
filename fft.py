import numpy as np#导入一个数据处理模块
import pylab as pl#导入一个绘图模块，matplotlib下的模块
from HI_data_prepare import signal_pca
import pywt

# csi = np.ones(800,270)

def get_csi_data_ph_dif(sample):  # 输入(50~~~, 270)

    datax = sample.T[0:90, :]  # (90,50~~~)
    data_phase_ori = np.angle(datax)
    data_phase = np.unwrap(data_phase_ori)
    row = int(data_phase.shape[0] / 3)  # 30
    colum = int(data_phase.shape[1])  # 50~~~
    data_phase_dif = np.ones([row, colum])
    for i in range(90):
        if (i < 30):
            data_phase_dif[i] = data_phase_ori[i + 29] - data_phase_ori[i]  # 默认为行之间相减
        # else:
        #    data_phase_dif[i] = data_phase_ori[i-60] - data_phase_ori[i]
    data_pca_ph = signal_pca(data_phase_dif.T, 2)
    return data_pca_ph  # (50~~~,)
# ss = get_csi_data_ph_dif(csi)
# print(1)
#
#
#
#
# def get_dwt(sample):
#     cso = abs(sample[:,0:2])
#     pl.subplot(411)
#     pl.plot(cso)
#     coeffs = pywt.wavedec2(cso,'db1', level = 2) #返回CA，（CH，CV，CD）分别是逼近，水平，垂直和对角线细节
#     pl.subplot(412)
#     pl.plot(coeffs[0])
#     pl.subplot(413)
#     pl.plot(coeffs[1][0])
#     pl.subplot(414)
#     pl.plot(coeffs[2][2])
#     ss = pywt.waverec2(coeffs, 'db1')
#     print(2)
#     # pl.subplot(213)
#     # pl.plot(ss)
#     pl.show()
#     return ss
# ss = get_dwt(csi[1,:,:])
# print(1)


















#
# def fft(csi):
#     x = csi[1,:,1]
#     sampling_rate = 8000#采样频率为8000Hz
#     fft_size = 1024 #FFT处理的取样长度
#     t = np.arange(0, 1.0, 1.0/sampling_rate)#np.arange(起点，终点，间隔)产生1s长的取样时间
#     # x = np.sin(2*np.pi*156.25*t) + 2*np.sin(2*np.pi*234.375*t)#两个正弦波叠加，156.25HZ和234.375HZ
#     # N点FFT进行精确频谱分析的要求是N个取样点包含整数个取样对象的波形。因此N点FFT能够完美计算频谱对取样对象的要求是n*Fs/N（n*采样频率/FFT长度），
#     # 因此对8KHZ和512点而言，完美采样对象的周期最小要求是8000/512=15.625HZ,所以156.25的n为10,234.375的n为15。
#     xs = x[:fft_size]# 从波形数据中取样fft_size个点进行运算
#     xf = np.fft.rfft(xs)/fft_size# 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
#     # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
#     #于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
#     freqs = np.linspace(0, sampling_rate//2, fft_size//2+1)
#     freqs = freqs[0:401]
#     # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#     #在指定的间隔内返回均匀间隔的数字
#     xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
#     #最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
#
#     #绘图显示结果
#     pl.figure(figsize=(8,4))
#     pl.subplot(211)
#     pl.plot(t[:800], xs)
#     pl.xlabel(u"Time(S)")
#     pl.title(u"156.25Hz and 234.375Hz WaveForm And Freq")
#     pl.subplot(212)
#     pl.plot(freqs, xfp)
#     pl.xlabel(u"Freq(Hz)")
#     pl.subplots_adjust(hspace=0.4)
#     pl.show()
#
# ss = fft(csi)
