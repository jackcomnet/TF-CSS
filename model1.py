import torch
from torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv
# from torchsummary import summary
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Origin_net(nn.Module):
    def __init__(self):
        super(Origin_net, self).__init__()
        self.conv1 = ComplexConv(in_channels=1, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.maxpool7 = nn.MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.maxpool8 = nn.MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.maxpool9 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)
        self.linear = nn.Linear(1024, 30)

        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)

        x = self.conv8(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)

        x = self.conv9(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = F.tanh(x)
        # x = self.dropout(x)
        # x = self.linear(x)
        return x


class Origin_classifier(nn.Module):
    def __init__(self):
        super(Origin_classifier, self).__init__()
        # self.conv10 = nn.Conv1d(in_channels=1024, out_channels=16384, kernel_size=1, stride=1)
        self.linear4 = nn.LazyLinear(1024)
        self.transposed_conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        self.transposed_conv2 = nn.ConvTranspose1d(in_channels=10, out_channels=10, kernel_size=3, stride=2)
        # self.dropout = nn.Dropout(0.3)
        self.flatten2 = nn.Flatten()
        self.linear2 = nn.LazyLinear(4116)

    def forward(self,x):
        y = self.linear4(x)
        y = F.sigmoid(y)

        x = torch.unsqueeze(y, dim=1)
        # x = self.conv10(x)
        x = self.transposed_conv1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.transposed_conv2(x)
        x = F.leaky_relu(x, 0.2)

        x =self.flatten2(x)
        # x = self.dropout(x)
        x =self.linear2(x)
        x = x.view(-1, 2, 2048)
        return y,x



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
            fft = x
            # fft = torch.abs(x)
        else:
            if (self.low_freq is None) and (self.high_freq is not None):
                mask = freqs <= self.high_freq
            elif (self.low_freq is not None) and (self.high_freq is None):
                mask = freqs >= self.low_freq
            else:
                mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
            fft = x[..., mask]
            # fft = torch.abs(x[..., mask])
            freqs = freqs[mask]

        if self.low_freq is None or self.low_freq < 0:  # 如果包含了负频率，标准化因子就不补偿2，而是1
            factor = 1.0
            # print(factor)
        return (factor / num_samples) * fft, freqs


# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values

# 定义自注意力分类器模型
class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        attended_values = self.attention(x)
        x = attended_values.mean(dim=1)  # 对每个位置的向量求平均
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MultiHead_SelfAttention(nn.Module):
    def __init__(self,input_dim,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=input_dim//num_heads#head的维度为输入维度除以head个数，方便后面拼接
        assert input_dim%num_heads==0 ,"Input dimension must be divisible by the number of heads."

        # Linear layers for the query, key, and value projections for each head
        self.query=nn.Linear(input_dim,input_dim)
        self.key=nn.Linear(input_dim,input_dim)
        self.value=nn.Linear(input_dim,input_dim)

        self.output_linear=nn.Linear(input_dim,input_dim)

    def forward(self,x):
        batch_size,seq_len,input_dim=x.size()
        #输入数据shape=[batch_size,token个数，token长度]
        #将输出向量经过矩阵乘法后拆分为多头
        query=self.query(x).view(batch_size,seq_len,self.num_heads,self.head_dim)
        #输入数据shape=[batch_size,token个数，head数，head维度]
        key=self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value=self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        #对调序列的长度和head个数(batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, seq_len, head_dim)
        #方便后续矩阵乘法和不同头部的注意力计算
        query=query.transpose(1,2)#(batch_size, num_heads, seq_len, head_dim)
        key=key.transpose(1,2)
        value=value.transpose(1,2)
        #计算注意力分数和权重matmul:最后两个维度做矩阵乘法
        attention_scores=torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float))
        #query:(batch_size, num_heads, seq_len, head_dim) * key(batch_size,num_heads,head_dim,seq_len)
        #attention_scores:(batch_size, num_heads, seq_len, seq_len)
        attention_weights=torch.softmax(attention_scores,dim=-1)
        #注意力加权求和
        attention=torch.matmul(attention_weights,value)
        # attention_scores:(batch_size, num_heads, seq_len, seq_len)* value(batch_size, num_heads, seq_len, head_dim)
        #attention:(batch_size, num_heads, seq_len, head_dim)
        #连接和线性变换
        attention=attention.transpose(1,2).contiguous().view(batch_size,seq_len,input_dim)#contiguos深拷贝，不改变原数据
        #(batch_size,num_heads, seq_len , head_dim) to (batch_size, seq_len, num_heads, head_dim) to (batch_size,seq_len,input_dim)
        output=self.output_linear(attention)

        return output

#定义多头自注意力机制模型
class MultiHead_SelfAttention_Classifier(nn.Module):
    def __init__(self,input_dim,num_heads,hidden_dim,num_classes):
        super().__init__()
        self.attention=MultiHead_SelfAttention(input_dim,num_heads)
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,num_classes)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.attention(x)
        x=x.mean(dim=1)#(batch_size, seq_len, input_dim) to (batch_size, input_dim)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

class ChannelAttention(nn.Module):#channel attention
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, another_x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return another_x * self.sigmoid(y)


class SpatialAttention(nn.Module):#spatial attention
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, another_x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]#沿着不同通道中的同一个位置计算，返回值和索引，取值
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return another_x * self.sigmoid(y)

class CBAMa(nn.Module):#Convolutional Block Attention Module
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMa, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x, x)
        y = self.spatial_attention(x, x)
        return y
class CBAMb(nn.Module):#Convolutional Block Attention Module
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMb, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x, another_x):
        y = self.channel_attention(x, another_x)
        y = self.spatial_attention(y, y)
        return y
# ----------------------------------------------------#
#   LSTM 模型
# ----------------------------------------------------#
class MyNET(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.lstm = nn.GRU(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers ,batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_classes)  # num_classes classes

        self.seqt1 = nn.Sequential(
            ComplexConv(in_channels=1, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.seqt2 = nn.Sequential(
            ComplexConv(in_channels=64, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.seqt3 = nn.Sequential(
            ComplexConv(in_channels=64, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.seqf1 = nn.Sequential(
            ComplexConv(in_channels=1, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.seqf2 = nn.Sequential(
            ComplexConv(in_channels=64, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.seqf3 = nn.Sequential(
            ComplexConv(in_channels=64, out_channels=64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
            ComplexConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.BN=nn.BatchNorm1d(num_features=256)
        self.fft = FFTLayer(sampling_rate=1)
        self.cbam1 = CBAMa(128, 1)
        self.cbam2 = CBAMa(128, 1)
        self.cbam3 = CBAMa(128, 1)
        self.cbam4 = CBAMa(128, 1)
        self.cbam5 = CBAMb(128, 1)
        self.cbam6 = CBAMb(128, 1)
        self.cbam7 = CBAMb(128, 1)
        self.cbam8 = CBAMb(128, 1)
        self.flatten = nn.Flatten()
        self.multihead_attention1 = MultiHead_SelfAttention(256,8)
        self.multihead_attention2 = MultiHead_SelfAttention(32, 1)
        self.multihead_attention3 = MultiHead_SelfAttention(256, 8)
        self.multihead_attention4 = MultiHead_SelfAttention(32, 1)
        self.multihead_attention5 = MultiHead_SelfAttention(128, 4)
        self.linear1 = nn.LazyLinear(2048)
        self.linear2 = nn.LazyLinear(1024)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(1024,num_classes)

        self.attention = SelfAttention(32)
        self.fc1 = nn.Linear(32, 30)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # data_complex = torch.zeros((x.shape[0], x.shape[2]), dtype=torch.complex64, device=device)
        # data_complex[:, :] = x[:, 0] + 1j * x[:, 1]  # 每个样本按列分成I路
        # data_fft=torch.fft.fft(data_complex)
        #
        # x_fft = torch.zeros((data_fft.shape[0], 2, data_fft.shape[1]), device=device)
        # x_fft[:, 0, :] = torch.real(data_fft[:, :])  # 每个样本按列分成I路
        # x_fft[:, 1, :] = torch.imag(data_fft[:, :])  # 每个样本按列分成Q路


        xt=x

        x = self.seqt1(x)
        # x = self.multihead_attention1(x)
        x = self.cbam1(x)
        x = self.seqt2(x)
        # x = self.multihead_attention2(x)
        x = self.cbam2(x)
        # x = self.seqt3(x)
        # x21 = self.seqt3(x2)


        xtc = torch.zeros(xt.shape[0], xt.shape[2], dtype=torch.complex64)
        xf = torch.zeros(xt.shape[0], xt.shape[1], xt.shape[2])
        xtc[:,:] = xt[:,0,:]+1j*xt[:,1,:]
        xtc = xtc.unsqueeze(1)
        fft, freq = self.fft(xtc)
        mask = (freq >= -0.1) & (freq <= 0.1)
        xf[:,0,:]=fft[:,0,:].real
        xf[:,1,:]=fft[:,0,:].imag

        xf = self.seqf1(xf.to(device))
        # xf = self.multihead_attention3(xf)
        xf = self.cbam3(xf)
        xf = self.seqf2(xf)
        # xf = self.multihead_attention4(xf)
        xf = self.cbam4(xf)
        # xf = self.seqf3(xf)
        xfc = torch.zeros(xf.shape[0], xf.shape[2], dtype=torch.complex64)
        xft = torch.zeros(xf.shape[0], xf.shape[1], xf.shape[2])
        xfc[:, :] = xf[:, 0, :] + 1j * xf[:, 1, :]
        xfc = xfc.unsqueeze(1)
        ifft = torch.fft.ifft(xfc)
        xft[:,0,:]=ifft[:,0,:].real
        xft[:,1,:]=ifft[:,0,:].imag

        # xf21 = self.seqf3(xf2)

        # x1 = self.cbam3(x, xf)
        # x2 = self.cbam4(xf, x)

##############################################

        # x1_fft, _ = self.lstm(x_fft_real)  # LSTM层
        # # x1_fft = x1_fft[:,-1, :]  # 只取LSTM输出中的最后一个时间步
        #
        # x2_fft, _ = self.lstm(x_fft_image)  # LSTM层
        # # x2_fft = x2_fft[:, -1, :]  # 只取LSTM输出中的最后一个时间步
        # x = torch.cat((x1_fft, x2_fft), dim=1)
        #
        # # x = torch.cat((x, x_fft), dim=1)




        x = torch.cat((x, xft.to(device)), dim=1)
        x = self.multihead_attention5(x)
        # x = x.mean(dim=1)  # (batch_size, seq_len, input_dim) to (batch_size, input_dim)
        # x = self.fc1(x)
        # xa=x
        #
        # x = self.BN(x)
        # attended_values = self.attention(x)
        # x = attended_values.mean(dim=1)  # 对每个位置的向量求平均
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = F.tanh(x)
        # x = self.fc2(x)

        # x = self.flatten(x)
        x = self.linear1(x)

        x = F.tanh(x)
        x = self.linear2(x)
        # x = self.dropout(x)
        x = F.tanh(x)
        # x = self.linear(x)  # 通过一个全连接层
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv10 = nn.Conv1d(in_channels=1024, out_channels=16384, kernel_size=1, stride=1)

        # self.linear1 = nn.LazyLinear(2048)

        # self.transposed_conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        # self.transposed_conv2 = nn.ConvTranspose1d(in_channels=10, out_channels=10, kernel_size=3, stride=2)
        # self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        # self.linear2 = nn.LazyLinear(4096)
    def forward(self,x):
        # x = self.linear1(x)
        x = torch.unsqueeze(x, dim=2)
        x = self.conv10(x)
        # x = self.transposed_conv1(x)
        x = F.tanh(x)
        # x = self.transposed_conv2(x)
        # x = F.tanh(x)
        # x =self.flatten(x)
        # x = self.dropout(x)
        # x =self.linear2(x)
        # x = F.tanh(x)
        x = x.view(-1, 2, 8192)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(1024,30)

    def forward(self,x):
        # x = self.dropout(x)
        x = self.linear(x)

        return x