import torch
from torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv
# from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=4,stride=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=4,stride=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)

        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2)

        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.maxpool7 = nn.MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.maxpool8 = nn.MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.maxpool9 = nn.MaxPool1d(kernel_size=2)

        self.conv10 = ComplexConv(in_channels=1, out_channels=64, kernel_size=4,stride=1)
        self.batchnorm10 = nn.BatchNorm1d(num_features=128)
        self.maxpool10 = nn.MaxPool1d(kernel_size=2)
        self.conv11 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm11 = nn.BatchNorm1d(num_features=128)
        self.maxpool11 = nn.MaxPool1d(kernel_size=2)
        self.conv12 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm12 = nn.BatchNorm1d(num_features=128)
        self.maxpool12 = nn.MaxPool1d(kernel_size=2)
        self.conv13 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm13 = nn.BatchNorm1d(num_features=128)
        self.maxpool13 = nn.MaxPool1d(kernel_size=2)

        self.conv14 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm14 = nn.BatchNorm1d(num_features=128)
        self.maxpool14 = nn.MaxPool1d(kernel_size=2)
        self.conv15 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm15 = nn.BatchNorm1d(num_features=128)
        self.maxpool15 = nn.MaxPool1d(kernel_size=2)

        self.conv16 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm16 = nn.BatchNorm1d(num_features=128)
        self.maxpool16 = nn.MaxPool1d(kernel_size=2)

        self.conv17 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm17 = nn.BatchNorm1d(num_features=128)
        self.maxpool17 = nn.MaxPool1d(kernel_size=2)
        self.conv18 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm18 = nn.BatchNorm1d(num_features=128)
        self.maxpool18 = nn.MaxPool1d(kernel_size=2)
        self.conv19 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm19 = nn.BatchNorm1d(num_features=128)
        self.maxpool19 = nn.MaxPool1d(kernel_size=2)

        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)
        self.linear2 = nn.LazyLinear(1024)

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.batchnorm1(x1)
        # x1 = self.maxpool1(x1)

        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm2(x1)
        # x1 = self.maxpool2(x1)

        x1 = self.conv3(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm3(x1)
        # x1 = self.maxpool3(x1)

        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm4(x1)
        # x1 = self.maxpool4(x1)

        x1 = self.conv5(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm5(x1)
        # x1 = self.maxpool5(x1)

        x1 = self.conv6(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm6(x1)
        x1 = self.maxpool6(x1)

        x1 = self.conv7(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm7(x1)
        x1 = self.maxpool7(x1)

        x1 = self.conv8(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm8(x1)
        x1 = self.maxpool8(x1)

        x1 = self.conv9(x1)
        x1 = F.relu(x1)
        x1 = self.batchnorm9(x1)
        x1 = self.maxpool9(x1)



        x2 = self.conv10(x)
        x2 = F.relu(x2)
        x2 = self.batchnorm10(x2)
        # x2 = self.maxpool10(x2)

        x2 = self.conv11(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm11(x2)
        # x2 = self.maxpool11(x2)

        x2 = self.conv12(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm12(x2)
        # x2 = self.maxpool12(x2)

        x2 = self.conv13(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm13(x2)
        # x2 = self.maxpool13(x2)

        x2 = self.conv14(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm14(x2)
        # x2 = self.maxpool14(x2)

        x2 = self.conv15(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm15(x2)
        x2 = self.maxpool15(x2)

        x2 = self.conv16(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm16(x2)
        x2 = self.maxpool16(x2)

        x2 = self.conv17(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm17(x2)
        x2 = self.maxpool17(x2)

        x2 = self.conv18(x2)
        x2 = F.relu(x2)
        x2 = self.batchnorm18(x2)
        x2 = self.maxpool18(x2)

        # x2 = self.conv19(x2)
        # x2 = F.relu(x2)
        # x2 = self.batchnorm19(x2)
        # x2 = self.maxpool19(x2)

        x1 = self.flatten1(x1)
        x1 = self.linear1(x1)
        x2 = self.flatten2(x2)
        x2 = self.linear2(x2)
        # x3 = self.linear11(self.flatten(x3))
        # x4 = self.linear11(self.flatten(x4))
        #
        # x=torch.cat([x1,x3,x4], dim=1)

        # x = self.linear1(x)

        return x1,x2



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
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
        x = x.view(-1, 2, 2058)
        return y,x

class Feature_Classifier(nn.Module):
    def __init__(self):
        super(Feature_Classifier, self).__init__()
        # self.conv10 = nn.Conv1d(in_channels=1024, out_channels=16384, kernel_size=1, stride=1)
        # self.transposed_conv3 = nn.ConvTranspose1d(in_channels=1, out_channels=10, kernel_size=3, stride=1)
        # self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.LazyLinear(1024)

    def forward(self,x):
        x = self.linear3(x)
        x = F.sigmoid(x)

        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear5 = nn.LazyLinear(1024)
        # self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(1024,30)

    def forward(self,x):
        y = self.linear5(x)
        y = F.sigmoid(y)

        # x = self.dropout(y)
        x = self.linear(y)

        return y,x


if __name__ == "__main__":
    x = torch.randn((10,2,8192))
    encoder = Encoder()
    decoder = Decoder()
    classifier = Classifier()
    z = encoder(x)
    x_r = decoder(z)
    cl = classifier(z)
    print(z.shape)
    print(x_r.shape)
    print(cl.shape)



