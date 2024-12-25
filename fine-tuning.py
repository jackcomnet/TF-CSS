import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from model1 import *
from get_fewshot_LoRa_IQ_dataset import *
from sklearn import metrics
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment  # 添加as语句不用修改代码中的函数名
import random
import pandas as pd
import os
from thop import profile, clever_format
import matplotlib.pyplot as plt

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


def train(encoder,
          classifier,
          dataloader,
          optim_encoder,
          optim_classifier,
          scheduler_encoder,
          scheduler_classifier,
          epoch,
          device_num,
          k_shot,
          writer
          ):  # 用编码器

    encoder.train()
    classifier.train()  # 训练模式
    device = torch.device("cuda:" + str(device_num))

    loss_ce, correct, total_correct_1, total_correct_5= 0.0, 0.0, 0.0, 0.0

    for data_label in dataloader:
        data, target = data_label
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optim_encoder.zero_grad()
        optim_classifier.zero_grad()

        z = encoder(data)
        data_r = classifier(z)

        logits = F.log_softmax(data_r)
        loss_ce_batch = F.nll_loss(logits, target)

        loss_ce_batch.backward()

        optim_encoder.step()
        optim_classifier.step()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss_ce += loss_ce_batch.item()

        prediction = torch.argsort(F.softmax(data_r, -1), dim=-1, descending=True)
        total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    loss_ce /= len(dataloader)

    scheduler_encoder.step()  # 每次迭代更新一次学习率
    scheduler_classifier.step()
    fmt = 'Train Epoch: {} \tLearning Rate: {:.10f} \tCE_Loss, {:.8f}, Accuracy: {}/{} ({:0f}%), ACC@1: {:.2f}% ACC@5: {:.2f}%\n'
    print(
        fmt.format(
            epoch,
            scheduler_encoder.get_last_lr()[0],
            loss_ce,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
            total_correct_1 / len(dataloader.dataset) * 100,
            total_correct_5 / len(dataloader.dataset) * 100,
        )
    )

    writer.add_scalar('CE_Loss/train', loss_ce, epoch)
    return loss_ce,100.0 * correct / len(dataloader.dataset)


def validation(encoder, classifier, test_dataloader, epoch, device_num, writer):  # 获得第epoch轮自动编码器的mse和其中编码器输入特征的sc分数
    encoder.eval()
    classifier.eval()

    loss_ce, correct, total_correct_1, total_correct_5= 0.0, 0.0, 0.0, 0.0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            z = encoder(data)
            data_r = classifier(z)
            logits = F.log_softmax(data_r)
            loss_ce += F.nll_loss(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            prediction = torch.argsort(F.softmax(data_r, -1), dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        loss_ce /= len(test_dataloader)

        fmt = '\nValidation set: CE_loss: {:.4f}, Accuracy: {}/{} ({:0f}%), ACC@1: {:.2f}% ACC@5: {:.2f}%\n'
        print(
            fmt.format(
                loss_ce,
                correct,
                len(test_dataloader.dataset),
                100.0 * correct / len(test_dataloader.dataset),
                total_correct_1 / len(test_dataloader.dataset) * 100,
                total_correct_5 / len(test_dataloader.dataset) * 100,
            )
        )
        writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
        writer.add_scalar('Classifier_Loss/validation', loss_ce, epoch)  # 画图


    return loss_ce


def Test(encoder, classifier, test_dataloader):
    encoder.eval()
    classifier.eval()

    test_loss, correct, total_correct_1, total_correct_5= 0.0, 0.0, 0.0, 0.0
    loss = nn.NLLLoss()
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    y_gt = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)

            z = encoder(data)

            data_r = classifier(z)

            logits = F.log_softmax(data_r)
            test_loss += loss(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            prediction = torch.argsort(F.softmax(data_r,-1), dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            # target_real[len(target_real):len(target) - 1] = target.tolist()

            # predict = pred.squeeze(dim=-1).cpu()
            # predict_np = predict.numpy()
            # targ = target.cpu()
            # targ_np = targ.numpy()
            # y_pred=np.append(y_pred,predict_np)
            # y_gt=np.append(y_gt,targ_np)

        # draw_confusion_matrix(label_true=y_gt, label_pred=y_pred, label_name=["0", "1", '2', '3', '4',
        #                                   '5', '6', '7', '8', '9', '10',
        #                                   '11', '12', '13', "14",
        #                                   "15", "16", "17", "18", "19",
        #                                   "20","21", "22", '23', '24', '25',
        #                                   '26', '27', '28', '29'],
        #                       title="Confusion Matrix on test dataset based on TF-CSS with {}-Shot".format(conf.k_shot),
        #                       pdf_save_path="confus_maxi/TF-CSS_{}.eps".format(conf.k_shot))

    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%), ACC@1: {:.2f}% ACC@5: {:.2f}%\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
            total_correct_1 / len(test_dataloader.dataset) * 100,
            total_correct_5 / len(test_dataloader.dataset) * 100,
        )
    )
    test_acc = 100.0 * correct / len(test_dataloader.dataset)
    acc1 = total_correct_1 / len(test_dataloader.dataset) * 100
    acc5 = total_correct_5 / len(test_dataloader.dataset) * 100
    return str(test_acc) + '%' , str(acc1) + '%' , str(acc5) + '%'

# def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=1000):
#     cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
#     plt.figure(figsize=(10, 8))
#     plt.imshow(cm, cmap='Blues')
#     plt.title(title)
#     plt.xlabel("Predict label")
#     plt.ylabel("Truth label")
#     plt.yticks(range(label_name.__len__()), label_name,fontsize=5)
#     plt.xticks(range(label_name.__len__()), label_name, rotation=90,fontsize=5)
#     plt.tight_layout()
#     plt.colorbar()
#     for i in range(label_name.__len__()):
#         for j in range(label_name.__len__()):
#             color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
#             value = float(format('%.2f' % cm[j, i]))
#             plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=5)
#     if not pdf_save_path is None:
#         plt.savefig(pdf_save_path, format='eps', bbox_inches='tight', dpi=dpi)

def train_and_validation(encoder,
                         classifier,
                         dataloader,
                         val_dataloader,
                         optim_encoder,
                         optim_classifier,
                         scheduler_encoder,
                         scheduler_classifier,
                         epochs,
                         encoder_save_path,
                         classifier_save_path,
                         device_num,
                         k_shot,
                         writer):
    train_loss = []
    train_acc = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            current_mse_dataloader = validation(encoder, classifier, dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数
            current_min_test_loss = validation(encoder, classifier, val_dataloader, epoch, device_num, writer)  # 获得对验证集的sc和mse分数

        loss,pre = train(encoder,
                               classifier,
                               dataloader,
                               optim_encoder,
                               optim_classifier,
                               scheduler_encoder,
                               scheduler_classifier,
                               epoch,
                               device_num,
                               k_shot,
                               writer)


        validation_loss = validation(encoder, classifier, val_dataloader, epoch, device_num, writer)

        if validation_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, validation_loss))
            current_min_test_loss = validation_loss
            torch.save(encoder, encoder_save_path)
            torch.save(classifier, classifier_save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

        train_loss.append(loss)  # 损失加入到列表中
        train_acc.append(pre)  # 准确率加入到列表中
    with open("train_loss/train_loss_{}.txt".format(k_shot), 'w') as train_los:
        train_los.write(str(train_loss))

    with open("train_acc/train_acc_{}.txt".format(k_shot), 'w') as train_ac:
        train_ac.write(str(train_acc))

class Config:
    def __init__(
            self,
            train_batch_size: int = 240,
            test_batch_size: int = 60,
            epochs: int = 500,
            lr_encoder: float = 0.002,
            lr_classifier: float = 0.002,
            n_classes: int = 30,
            encoder_save_path: str = 'model_weight/train_MAE_encoder_IQ.pth',
            classifier_save_path: str = 'model_weight/MAE_classifier_IQ.pth',
            encoder_load_path: str = 'model_weight/pretrain_MAE_encoder_IQ.pth',
            device_num: int = 0,
            iteration: int = 100,
            k_shot: int = 10,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_encoder = lr_encoder
        self.lr_classifier = lr_classifier
        self.n_classes = n_classes
        self.encoder_save_path = encoder_save_path
        self.classifier_save_path = classifier_save_path
        self.encoder_load_path = encoder_load_path
        self.device_num = device_num
        self.iteration = iteration
        self.k_shot = k_shot


def main(RANDOM_SEED):
    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs")

    set_seed(RANDOM_SEED)

    X_train, X_val, Y_train, Y_val = get_num_class_Targettrainfinetunedata(conf.n_classes, conf.k_shot)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))  # 训练集
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))  # 验证集
    val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True)

    encoder = torch.load(conf.encoder_load_path)  # 载入预训练好的编码器网络

    # encoder = encoder(n_features=32, hidden_size=32, num_layers=3, num_classes=conf.n_classes)
    classifier = Classifier()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    model_combined = nn.Sequential(
        encoder,
        classifier
    )
    flops, params = profile(model_combined, inputs=(torch.randn(1, 2, 8192).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    # optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr_encoder)
    # optim_origin_classifier = torch.optim.Adam(origin_classifier.parameters(), lr=conf.lr_origin_classifier)
    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr_encoder)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=conf.lr_classifier)

    # scheduler_encoder = CosineAnnealingLR(optim_encoder, T_max=20)  # 余弦退火调整学习率
    # scheduler_classifier = CosineAnnealingLR(optim_decoder, T_max=20)
    # scheduler_encoder = ExponentialLR(optim_encoder, gamma=0.9)
    # scheduler_classifier = ExponentialLR(optim_classifier, gamma=0.9)
    scheduler_encoder = StepLR(optim_encoder, step_size=4, gamma=0.95)
    scheduler_classifier = StepLR(optim_classifier, step_size=4, gamma=0.95)
    train_and_validation(encoder,
                         classifier,
                         train_dataloader,
                         val_dataloader,
                         optim_encoder,
                         optim_classifier,
                         scheduler_encoder,
                         scheduler_classifier,
                         conf.epochs,
                         conf.encoder_save_path,
                         conf.classifier_save_path,
                         conf.device_num,
                         conf.k_shot,
                         writer)

def Test_main():
    num = [0, 30]
    X_test, Y_test = get_num_class_Targettestdata(num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    encoder = torch.load(conf.encoder_save_path)
    classifier = torch.load(conf.classifier_save_path)
    test_acc,acc1,acc5 = Test(encoder, classifier, test_dataloader)
    return test_acc,acc1,acc5


if __name__ == '__main__':
    conf = Config()
    test_acc_all = []

    for i in range(conf.iteration):  # 所有epochs轮训练完算一次迭代
        print(f"iteration: {i}-------------------------------------------")
        main(i)
        test_acc,acc1,acc5 = Test_main()
        test_acc_all.append(test_acc)
    #     print(f"iteration={i},test_acc={test_acc}\n")
    # df = pd.DataFrame(test_acc_all)
    # df.to_excel(f"test_result/AMAE_{conf.n_classes}classes_{conf.k_shot}shot_{conf.iteration}iteration.xlsx")
