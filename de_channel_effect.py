import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from model1 import MyLSTM
from get_fewshot_LoRa_IQ_dataset1 import *
from sklearn import metrics
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment  # 添加as语句不用修改代码中的函数名
import random
import pandas as pd
import os
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现



lem=10

# 定义L1正则化函数
def l1_regularizer(weight, lambda_l1):
    return lambda_l1 * torch.norm(weight, 1)

# 定义L2正则化函数
def l2_regularizer(weight, lambda_l2):
    return lambda_l2 * torch.norm(weight, 2)



def l1_regularization(model):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.abs(param).sum()
    return 1e-5 * reg_loss

def pre_train(mylstm,
              dataloader,
              mask_ratio,
              optim_mylstm,
              scheduler_mylstm,
              epoch,
              device_num,
              writer
              ):  # 用编码器和解码器预训练
    mylstm.train()
    device = torch.device("cuda:" + str(device_num))
    loss_mse = 0
    correct = 0
    aum_size=100
    L=11
    for data_label in dataloader:
        data_channel, data = data_label
        if torch.cuda.is_available():
            data = data.to(device)
            data_channel = data_channel.to(device)
        optim_mylstm.zero_grad()

        data_r = mylstm(data_channel)
        loss_mse_batch = F.mse_loss(data_r, data)
        loss_mse_batch.backward()
        optim_mylstm.step()
        loss_mse += loss_mse_batch.item()

    loss_mse /= len(dataloader)
    scheduler_mylstm.step()  # 每次迭代更新一次学习率

    fmt = 'Train Epoch: {} \tCE_Loss, {:.8f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            epoch,
            loss_mse,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )

    writer.add_scalar('MSE_Loss/train', loss_mse, epoch)
    return loss_mse


def validation(mylstm, test_dataloader, epoch, device_num, writer):  # 获得第epoch轮自动编码器的mse和其中编码器输入特征的sc分数
    mylstm.eval()
    sc = 0
    loss_mse = 0
    correct = 0

    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data_channel, data in test_dataloader:
            if torch.cuda.is_available():
                data = data.to(device)
                data_channel = data_channel.to(device)

            data_r = mylstm(data_channel)

            loss_mse += F.mse_loss(data_r, data).item()


    loss_mse /= len(test_dataloader)
    fmt = '\nValidation set: MSE_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            loss_mse,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/validation', loss_mse, epoch) # 画图

    # X_test_embedding_feature_map, real_target = obtain_embedding_feature_map(encoder, test_dataloader)#获得测试集在编码器的输出特征
    # tsne = TSNE(n_components=2)#降到2维空间
    # eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    # km = KMeans(n_clusters=30, n_init=30)
    # km.fit(eval_tsne_embeds)
    # cluster_target = km.predict(eval_tsne_embeds)
    # sc = metrics.silhouette_score(X_test_embedding_feature_map, cluster_target)#-1到1之间，越大越好
    #
    # fmt = '\nValidation set: SC: {:.8f}\n'
    # print(
    #     fmt.format(
    #     sc,
    #     )
    # )

    # writer.add_scalar('SC/validation', sc, epoch)
    # df = pd.DataFrame(loss_mse)
    # df.to_excel(f"test_result/SimMIM_S_MSE.xlsx")
    return loss_mse


def train_and_validation(mylstm,
                         dataloader,
                         mask_ratio,
                         val_dataloader,
                         optim_mylstm,
                         scheduler_mylstm,
                         epochs,
                         mylstm_save_path,
                         device_num,
                         writer):
    current_sc_mse = 0
    gamma = 1
    loss_sc_mse_all = []
    loss_mse_all = []
    loss_sc_all = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            current_mse_dataloader = validation(mylstm, dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            current_mse = validation(mylstm, val_dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            # current_sc_mse = current_sc - current_mse
        train_mse_loss = pre_train(mylstm,
                                   dataloader,
                                   mask_ratio,
                                   optim_mylstm,
                                   scheduler_mylstm,
                                   epoch,
                                   device_num,
                                   writer)
        mse = validation(mylstm, val_dataloader, epoch, device_num, writer)

        # sc_mse = sc - 0.3 * mse
        # loss_sc_mse_all.append(sc_mse)
        loss_mse_all.append(mse)
        # loss_sc_all.append(sc)

        if mse < current_mse:
            print("The training SC-MSE is improved from {} to {}, new model weight is saved.".format(
                current_mse, mse))
            current_mse = mse
            torch.save(mylstm, mylstm_save_path)

        else:
            print("The training SC-MSE is not improved.")
        print("------------------------------------------------")

        writer.add_scalar('UnsupervisedLoss/train', mse, epoch)

        # torch.save(encoder, encoder_save_path)
    # df = pd.DataFrame(loss_sc_mse_all)
    # df.to_excel(f"test_result/SimMIM_S_SC_MSE.xlsx")
    df = pd.DataFrame(loss_mse_all)
    df.to_excel(f"test_result/SimMIM_S_MSE.xlsx")
    # df = pd.DataFrame(loss_sc_all)
    # df.to_excel(f"test_result/SimMIM_S_SC.xlsx")


class Config:
    def __init__(
            self,
            train_batch_size: int = 256,
            test_batch_size: int = 256,
            epochs: int = 300,
            mask_ratio: float = 0.3,
            lr_mylstm: float = 0.001,
            n_classes: int = 30,
            mylstm_save_path: str = 'model_weight/train_MAE_mylstm_IQ.pth',
            device_num: int = 0,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.mask_ratio = mask_ratio
        self.lr_mylstm = lr_mylstm
        self.n_classes = n_classes
        self.mylstm_save_path = mylstm_save_path
        self.device_num = device_num


def main():
    # 创建两个信号
    complex_data = np.array([1+1j*2, 2+1j*2, 3+1j*3, 4+1j*4])
    signal_b = np.array([10+1j*2, 20+1j*2, 30+1j*2, 40+1j*2])

    # 使用NumPy的convolve函数进行卷积
    convolved_result = np.convolve(complex_data,signal_b)

    print(convolved_result)

    # print(np.random.randn(1, 10))

    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs")
    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    aum_size=100

    data_after_convolved, X_train, data_after_convolved_val, X_val, label_convolved, Y_train, label_convolved_val, Y_val = get_num_class_Sourcetraindata(conf.n_classes)
    X_train_enlarge = np.empty((X_train.shape[0] * aum_size, 2, X_train.shape[2]))
    for i in range(len(X_train)):
        for k in range(0, aum_size):
            X_train_enlarge[aum_size * i + k] = X_train[i]

    X_val_enlarge = np.empty((X_val.shape[0] * aum_size, 2, X_val.shape[2]))
    for i in range(len(X_val)):
        for k in range(0, aum_size):
            X_val_enlarge[aum_size * i + k] = X_val[i]


    train_dataset = TensorDataset(torch.Tensor(data_after_convolved), torch.Tensor(X_train_enlarge))  # 训练集
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)
    # for data, target in train_dataloader:
    #     print(data)
    #     print(target)
    val_dataset = TensorDataset(torch.Tensor(data_after_convolved_val), torch.Tensor(X_val_enlarge))  # 验证集
    val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True)

    mylstm = MyLSTM(n_features=32, hidden_size=32, num_layers=3, num_classes=conf.n_classes)

    if torch.cuda.is_available():
        mylstm = mylstm.to(device)


    # optim_mylstm = torch.optim.Adam(mylstm.parameters(), lr=conf.lr_mylstm)
    # optim_origin_classifier = torch.optim.Adam(origin_classifier.parameters(), lr=conf.lr_origin_classifier)
    optim_mylstm = torch.optim.Adam(mylstm.parameters(), lr=conf.lr_mylstm)

    # scheduler_encoder = CosineAnnealingLR(optim_encoder, T_max=20)  # 余弦退火调整学习率
    # scheduler_classifier = CosineAnnealingLR(optim_decoder, T_max=20)
    scheduler_mylstm = ExponentialLR(optim_mylstm, gamma=0.9)
    train_and_validation(mylstm,
                         train_dataloader,
                         conf.mask_ratio,
                         val_dataloader,
                         optim_mylstm,
                         scheduler_mylstm,
                         conf.epochs,
                         conf.mylstm_save_path,
                         conf.device_num,
                         writer)


if __name__ == '__main__':
    main()
