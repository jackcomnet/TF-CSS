import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from model1 import Origin_net, Origin_classifier
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

def pre_train(origin_net,
              origin_classifier,
              dataloader,
              mask_ratio,
              optim_origin_net,
              optim_origin_classifier,
              scheduler_origin_net,
              scheduler_origin_classifier,
              epoch,
              device_num,
              writer
              ):  # 用编码器和解码器预训练
    origin_net.train()
    origin_classifier.train()
    device = torch.device("cuda:" + str(device_num))
    loss_ce = 0
    correct = 0
    for data_label in dataloader:
        data, target = data_label
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optim_origin_net.zero_grad()
        optim_origin_classifier.zero_grad()
        data_r = origin_net(data)
        # data_r = origin_classifier(z1)

        logits = F.log_softmax(data_r)
        target = np.squeeze(target, axis=1)
        loss_ce_batch = F.nll_loss(logits, target)

        # 定义L1和L2正则化参数
        lambda_l1 = 0.01
        lambda_l2 = 0.01

        # 4.2 计算L1和L2正则化
        # l1_regularization_origin_net = l1_regularizer(origin_net.parameters(), lambda_l1)
        # l2_regularization_origin_net = l2_regularizer(origin_net.parameters(), lambda_l2)
        # l1_regularization_origin_classifier = l1_regularizer(origin_classifier.parameters(), lambda_l1)
        # l2_regularization_origin_classifier = l2_regularizer(origin_classifier.parameters(), lambda_l2)
        # 4.3 向loss中加入L1和L2
        # loss_ce_batch += l1_regularization(origin_net)


        loss_ce_batch.backward()

        optim_origin_net.step()
        optim_origin_classifier.step()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss_ce += loss_ce_batch.item()

    loss_ce /= len(dataloader)
    scheduler_origin_net.step()  # 每次迭代更新一次学习率
    scheduler_origin_classifier.step()  # 每次迭代更新一次学习率

    fmt = 'Train Epoch: {} \tCE_Loss, {:.8f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            epoch,
            loss_ce,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )

    writer.add_scalar('CE_Loss/train', loss_ce, epoch)
    return loss_ce


def validation(origin_net, origin_classifier, test_dataloader, epoch, device_num, writer):  # 获得第epoch轮自动编码器的mse和其中编码器输入特征的sc分数
    origin_net.eval()
    origin_classifier.eval()
    sc = 0
    loss_ce = 0
    correct = 0
    fc = []
    lp = []
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            data_r = origin_net(data)
            # data_r = origin_classifier(z1)
            logits = F.log_softmax(data_r)
            target = np.squeeze(target, axis=1)
            loss_ce += F.nll_loss(logits, target).item()

            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss_ce /= len(test_dataloader)
    fmt = '\nValidation set: CE_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            loss_ce,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('Classifier_Loss/validation', loss_ce, epoch) # 画图

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
    return loss_ce


def train_and_validation(origin_net,
                         origin_classifier,
                         dataloader,
                         mask_ratio,
                         val_dataloader,
                         optim_origin_net,
                         optim_origin_classifier,
                         scheduler_origin_net,
                         scheduler_origin_classifier,
                         epochs,
                         origin_net_save_path,
                         origin_classifier_save_path,
                         device_num,
                         writer):
    current_sc_mse = 0
    gamma = 1
    loss_sc_mse_all = []
    loss_mse_all = []
    loss_sc_all = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            current_mse_dataloader = validation(origin_net, origin_classifier, dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            current_mse = validation(origin_net, origin_classifier, val_dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            # current_sc_mse = current_sc - current_mse
        train_mse_loss = pre_train(origin_net,
                                   origin_classifier,
                                   dataloader,
                                   mask_ratio,
                                   optim_origin_net,
                                   optim_origin_classifier,
                                   scheduler_origin_net,
                                   scheduler_origin_classifier,
                                   epoch,
                                   device_num,
                                   writer)
        mse = validation(origin_net, origin_classifier, val_dataloader, epoch, device_num, writer)

        # sc_mse = sc - 0.3 * mse
        # loss_sc_mse_all.append(sc_mse)
        loss_mse_all.append(mse)
        # loss_sc_all.append(sc)

        if mse < current_mse:
            print("The training SC-MSE is improved from {} to {}, new model weight is saved.".format(
                current_mse, mse))
            current_mse = mse
            torch.save(origin_net, origin_net_save_path)
            torch.save(origin_classifier, origin_classifier_save_path)

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
            lr_origin_net: float = 0.001,
            lr_origin_classifier: float = 0.001,
            n_classes: int = 30,
            origin_net_save_path: str = 'model_weight/train_MAE_origin_net_IQ.pth',
            origin_classifier_save_path: str = 'model_weight/train_MAE_origin_classifier_IQ.pth',
            device_num: int = 0,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.mask_ratio = mask_ratio
        self.lr_origin_net = lr_origin_net
        self.lr_origin_classifier = lr_origin_classifier
        self.n_classes = n_classes
        self.origin_net_save_path = origin_net_save_path
        self.origin_classifier_save_path = origin_classifier_save_path
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

    X_train, X_val, Y_train, Y_val = get_num_class_Sourcetraindata(conf.n_classes)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))  # 训练集
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)
    # for data, target in train_dataloader:
    #     print(data)
    #     print(target)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))  # 验证集
    val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True)

    origin_net = Origin_net()
    origin_classifier = Origin_classifier()

    if torch.cuda.is_available():
        origin_net = origin_net.to(device)
        origin_classifier = origin_classifier.to(device)


    # optim_origin_net = torch.optim.Adam(origin_net.parameters(), lr=conf.lr_origin_net)
    # optim_origin_classifier = torch.optim.Adam(origin_classifier.parameters(), lr=conf.lr_origin_classifier)
    optim_origin_net = torch.optim.Adam(origin_net.parameters(), lr=conf.lr_origin_net, weight_decay=0.0001)
    optim_origin_classifier = torch.optim.Adam(origin_classifier.parameters(), lr=conf.lr_origin_classifier, weight_decay=0.0001)

    # scheduler_encoder = CosineAnnealingLR(optim_encoder, T_max=20)  # 余弦退火调整学习率
    # scheduler_classifier = CosineAnnealingLR(optim_decoder, T_max=20)
    scheduler_origin_net = ExponentialLR(optim_origin_net, gamma=0.1)
    scheduler_origin_classifier = ExponentialLR(optim_origin_classifier, gamma=0.1)
    train_and_validation(origin_net,
                         origin_classifier,
                         train_dataloader,
                         conf.mask_ratio,
                         val_dataloader,
                         optim_origin_net,
                         optim_origin_classifier,
                         scheduler_origin_net,
                         scheduler_origin_classifier,
                         conf.epochs,
                         conf.origin_net_save_path,
                         conf.origin_classifier_save_path,
                         conf.device_num,
                         writer)


if __name__ == '__main__':
    main()
