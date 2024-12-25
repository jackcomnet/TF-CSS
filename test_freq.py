import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# 设置信号参数
num_channels = 5   # 信号通道数
num_samples = 600  # 时间样本点
sampling_rate = 200  # 采样率
low_freq = 0  # 选择想要的频率范围下限
high_freq = 80  # # 选择想要的频率范围上限【注：这里并不是滤波】

# 构造一个多通道信号
t = np.linspace(start=0, stop=(num_samples-1)/sampling_rate, num=num_samples)  # 时间轴
signals = 3*np.sin(2*np.pi*3*t)+2*np.sin(2*np.pi*15*t)+np.sin(2*np.pi*27*t)  # 包含三个分量，幅值分别为3,2,1，频率分别为3,15,27
noise_level = 0.5
noise = np.random.normal(0, noise_level, signals.shape)
signals = signals + noise  # 加入噪声
signals = np.tile(signals, (num_channels, 1))  # 复制成多通道信号
print('signals shape:', signals.shape)  # (channels,time_samples)

# 转换为tensor格式信号
tensor_signals = torch.tensor(signals, dtype=torch.float32)
# 使用PyTorch进行FFT
fft_signals = torch.fft.fft(tensor_signals, dim=1)
rfft_signals = torch.fft.rfft(tensor_signals, dim=1)
print('fft_signals shape:', fft_signals.shape)
print('rfft_signals shape:', rfft_signals.shape)

# 获取频率列表
freqs = torch.fft.fftfreq(num_samples, 1/sampling_rate)  # 得到全部频率
rfreqs = torch.fft.rfftfreq(num_samples, 1/sampling_rate)  # 得到非负频率
print('freqs length:', len(freqs))
print('freqs list:\n', freqs)
print('rfreqs length:', len(rfreqs))
print('rfreqs list:\n', rfreqs)

# 后续步骤使用freqs
# 选择想要的频率范围
mask = (freqs >= low_freq) & (freqs <= high_freq)

# 选择一个通道的信号进行可视化
channel_idx = 0
selected_channel_signal = signals[channel_idx]  # 时域信号
selected_channel_fft = fft_signals[channel_idx][mask]  # 选择频率范围内的频域信号

# 绘制原始信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, selected_channel_signal)
plt.title(f'Original Signal of Channel {channel_idx+1}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 绘制频域信号
plt.subplot(2, 1, 2)
plt.plot(freqs[mask], 2.0/num_samples * torch.abs(selected_channel_fft))  # 2.0/num_samples因子用于标准化
plt.title(f'FFT of Channel {channel_idx+1} ({low_freq}-{high_freq} Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()


# 自定义FFT层
class FFTLayer(nn.Module):
    """
    in: (batch,channel,time)
    out: (bath,channel,freq_len), (freq_len,)
    """

    def __init__(self, sampling_rate, low_freq=None, high_freq=None):
        super().__init__()
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate

    def forward(self, x):
        num_samples = x.shape[2]
        x = torch.fft.fft(x, dim=2)
        factor = 2.0
        freqs = torch.fft.fftfreq(num_samples, 1 / self.sampling_rate)

        if (self.low_freq is None) and (self.high_freq is None):
            fft = torch.abs(x)
        else:
            if (self.low_freq is None) and (self.high_freq is not None):
                mask = freqs <= self.high_freq
            elif (self.low_freq is not None) and (self.high_freq is None):
                mask = freqs >= self.low_freq
            else:
                mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
            fft = torch.abs(x[..., mask])
            freqs = freqs[mask]

        if self.low_freq is None or self.low_freq < 0:  # 如果包含了负频率，标准化因子就不补偿2，而是1
            factor = 1.0
            print(factor)
        return (factor / num_samples) * fft, freqs

sig = tensor_signals.unsqueeze(0)  # 将(channels,time_samples)转换为(1,channels,time_samples)，其中，1代表batch

fft, freq = FFTLayer(sampling_rate=sampling_rate)(sig)
print(fft.shape)
print(freq.shape)

# 绘制原始信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, sig[0, channel_idx, :])
plt.title(f'Original Signal of Channel {channel_idx+1}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 绘制频域信号
plt.subplot(2, 1, 2)
plt.plot(freq, fft[0, channel_idx, :])
plt.title(f'FFT of Channel {channel_idx+1} ({low_freq}-{high_freq} Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
