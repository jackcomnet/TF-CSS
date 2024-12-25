import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 创建一个简单的数据集
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.Tensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])


# 定义 LSTM 网络
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏层状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化细胞状态

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层
        return out


# 超参数
input_size = 10  # 输入维度（特征数量）
hidden_size = 128  # LSTM 隐藏层大小
num_layers = 2  # LSTM 层数
num_classes = 2  # 类别数
learning_rate = 0.001
num_epochs = 10
batch_size = 16

# 创建一些示例数据
sequences = [torch.randn(5, input_size) for _ in range(100)]  # 100 个序列，每个序列长度为 5，特征维度为 input_size
labels = [0 if i < 50 else 1 for i in range(100)]  # 50 个 0 类，50 个 1 类

# 创建数据集和数据加载器
dataset = SequenceDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (seqs, lbls) in enumerate(dataloader):
        seqs, lbls = seqs.to(device), lbls.to(device).squeeze()

        # 前向传播
        outputs = model(seqs)
        loss = criterion(outputs, lbls)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试
with torch.no_grad():
    correct = 0
    total = 0
    for seqs, lbls in dataloader:
        seqs, lbls = seqs.to(device), lbls.to(device).squeeze()
        outputs = model(seqs)
        _, predicted = torch.max(outputs.data, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


class ChannelAttention(nn.Module):#channel attention
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):#spatial attention
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]#沿着不同通道中的同一个位置计算，返回值和索引，取值
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        # x = x.view(x.size(0), -1)
        x = self.spatial_attention(x)
        return x


# 创建模型实例并加载预训练权重
model = MyModel()

# 设置示例输入
input = torch.randn(1, 3, 32, 32)

# 将模型导出为 ONNX 格式
torch.onnx.export(model, input, './model/Test/onnx_model.onnx')  # 导出后 netron.start(path) 打开

a = torch.randn(3, 4,2,5)
c = torch.randn(3, 4,1,1)
# print(a*c)
# print(a)
# b=nn.functional.softmax(a)
# print(b)
# a=nn.functional.softmax(a, dim=-1)
# print(a)

cbam=CBAM(4,1)
a=cbam(a)