import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from model1 import  *
# from model import Encoder, Decoder, Classifier
from get_fewshot_LoRa_IQ_dataset import *
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


def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            # target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                # target = target.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output) - 1] = output.tolist()  # 将NumPy数组转换为Python列表
            target_output[len(target_output):len(target) - 1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output


def get_accuracy_score(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1  # 组成混淆矩阵

    ind = linear_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    print('ACC = ', acc)
    return acc
lem=10

def pre_train(encoder,
              decoder,
              dataloader,
              mask_ratio,
              optim_encoder,
              optim_decoder,
              scheduler_encoder,
              scheduler_classifier,
              epoch,
              device_num,
              writer
              ):  # 用编码器和解码器预训练
    encoder.train()
    decoder.train()
    device = torch.device("cuda:" + str(device_num))
    loss_mse = 0

    fc = []
    lp = []
    for data_label in dataloader:
        data, target = data_label
        if torch.cuda.is_available():
            data = data.to(device)

        optim_encoder.zero_grad()
        optim_decoder.zero_grad()
        bbx1, bbx2, maskdata = MaskData(data, mask_ratio)
        z = encoder(maskdata)

        data_r = decoder(z)

        mask = torch.zeros((data.shape[0], data.shape[1], data.shape[2])).cuda()
        mask[:, :, bbx1: bbx2] = torch.ones((data.size()[1], bbx2 - bbx1)).cuda()
        data = data.mul(mask)
        data_r = data_r.mul(mask)

        loss_mse_batch = F.mse_loss(data_r, data)# 默认 L2 Normfc#只对掩码的部分计算mse
        # loss_mse_batch = F.mse_loss(data_r, data, reduction='sum') / (bbx2-bbx1)
        loss_mse_batch.backward()

        optim_encoder.step()
        optim_decoder.step()

        loss_mse += loss_mse_batch.item()

    loss_mse /= len(dataloader)
    scheduler_encoder.step()  # 每次迭代更新一次学习率
    scheduler_classifier.step()  # 每次迭代更新一次学习率

    print('Train Epoch: {} \tLearning Rate: {:.10f} \tMSE_Loss, {:.8f}\n'.format(
        epoch,
        scheduler_classifier.get_last_lr()[0],
        loss_mse,
    )
    )

    writer.add_scalar('MSE_Loss/train', loss_mse, epoch)
    return loss_mse


def validation(encoder, decoder, test_dataloader, epoch, device_num, writer):  # 获得第epoch轮自动编码器的mse和其中编码器输入特征的sc分数
    encoder.eval()
    decoder.eval()
    sc = 0
    loss_mse = 0

    fc = []
    lp = []
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            if torch.cuda.is_available():
                data = data.to(device)
            z= encoder(data)
            data_r = decoder(z)

            loss_mse += F.mse_loss(data_r, data).item()

    loss_mse /= len(test_dataloader)
    fmt = '\nValidation set: MSE loss: {:.8f}\n'
    print(
        fmt.format(
            loss_mse,
        )
    )

    writer.add_scalar('MSE_Loss/validation', loss_mse, epoch)  # 画图

    X_test_embedding_feature_map, real_target = obtain_embedding_feature_map(encoder, test_dataloader)#获得测试集在编码器的输出特征
    tsne = TSNE(n_components=2)#降到2维空间
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    km = KMeans(n_clusters=30, n_init=30)
    km.fit(eval_tsne_embeds)
    cluster_target = km.predict(eval_tsne_embeds)
    sc = metrics.silhouette_score(X_test_embedding_feature_map, cluster_target)#-1到1之间，越大越好

    fmt = '\nValidation set: SC: {:.8f}\n'
    print(
        fmt.format(
        sc,
        )
    )

    # writer.add_scalar('SC/validation', sc, epoch)
    # df = pd.DataFrame(loss_mse)
    # df.to_excel(f"test_result/SimMIM_S_MSE.xlsx")
    return sc, loss_mse


def train_and_validation(encoder,
                         decoder,
                         dataloader,
                         mask_ratio,
                         val_dataloader,
                         optim_encoder,
                         optim_decoder,
                         scheduler_encoder,
                         scheduler_classifier,
                         epochs,
                         encoder_save_path,
                         decoder_save_path,
                         device_num,
                         writer):
    current_sc_mse = 0
    gamma = 1
    loss_sc_mse_all = []
    loss_mse_all = []
    loss_sc_all = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            # current_mse_dataloader = validation(encoder, decoder, dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            current_sc, current_mse = validation(encoder, decoder, val_dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            current_sc_mse = current_sc - current_mse
        train_mse_loss = pre_train(encoder,
                                   decoder,
                                   dataloader,
                                   mask_ratio,
                                   optim_encoder,
                                   optim_decoder,
                                   scheduler_encoder,
                                   scheduler_classifier,
                                   epoch,
                                   device_num,
                                   writer)
        sc, mse = validation(encoder, decoder, val_dataloader, epoch, device_num, writer)

        sc_mse = sc - 0.3 * mse
        loss_sc_mse_all.append(sc_mse)
        loss_mse_all.append(mse)
        loss_sc_all.append(sc)

        if sc_mse > current_sc_mse:
            print("The training SC-MSE is improved from {} to {}, new model weight is saved.".format(
                current_sc_mse, sc_mse))
            current_sc_mse = sc_mse
            torch.save(encoder, encoder_save_path)
            torch.save(decoder, decoder_save_path)

        else:
            print("The training SC-MSE is not improved.")
        print("------------------------------------------------")

        writer.add_scalar('UnsupervisedLoss/train', mse, epoch)

        # torch.save(encoder, encoder_save_path)
    # df = pd.DataFrame(loss_sc_mse_all)
    # df.to_excel(f"test_result/SimMIM_S_SC_MSE.xlsx")
    # df = pd.DataFrame(loss_mse_all)
    # df.to_excel(f"test_result/SimMIM_S_MSE.xlsx")
    # df = pd.DataFrame(loss_sc_all)
    # df.to_excel(f"test_result/SimMIM_S_SC.xlsx")


class Config:
    def __init__(
            self,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            epochs: int = 300,
            mask_ratio: float = 0.3,
            lr_encoder: float = 0.001,
            lr_decoder: float = 0.001,
            n_classes: int = 30,
            encoder_save_path: str = 'model_weight/pretrain_MAE_encoder_IQ.pth',
            decoder_save_path: str = 'model_weight/pretrain_MAE_decoder_IQ.pth',
            device_num: int = 0,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.mask_ratio = mask_ratio
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.n_classes = n_classes
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path
        self.device_num = device_num


def main():
    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs")
    RANDOM_SEED = 0  # any random number
    set_seed(RANDOM_SEED)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    X_train, X_val, Y_train, Y_val = get_num_class_Sourcetraindata(conf.n_classes)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))  # 训练集
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)
    # for data, target in train_dataloader:
    #     print(data)
    #     print(target)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))  # 验证集
    val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True)
    # encoder = Origin_net()

    encoder = MyNET(num_classes=conf.n_classes)
    decoder = Decoder()

    if torch.cuda.is_available():
        encoder = encoder.to(device)
        decoder = decoder.to(device)


    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr_encoder)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=conf.lr_decoder)

    # scheduler_encoder = CosineAnnealingLR(optim_encoder, T_max=20)  # 余弦退火调整学习率
    # scheduler_classifier = CosineAnnealingLR(optim_decoder, T_max=20)
    # scheduler_encoder = ExponentialLR(optim_encoder, gamma=0.9)
    # scheduler_classifier = ExponentialLR(optim_decoder, gamma=0.9)
    scheduler_encoder = StepLR(optim_encoder, step_size=4, gamma=0.95)
    scheduler_classifier = StepLR(optim_decoder, step_size=4, gamma=0.95)
    train_and_validation(encoder,
                         decoder,
                         train_dataloader,
                         conf.mask_ratio,
                         val_dataloader,
                         optim_encoder,
                         optim_decoder,
                         scheduler_encoder,
                         scheduler_classifier,
                         conf.epochs,
                         conf.encoder_save_path,
                         conf.decoder_save_path,
                         conf.device_num,
                         writer)


if __name__ == '__main__':
    main()
