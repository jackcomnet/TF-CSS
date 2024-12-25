import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
from scipy import signal
import math
import json
import h5py

np.set_printoptions(threshold=np.inf)#所有np array数据写入txt


def complex_convert_to_I_Q_real(data):

    data_complex = np.zeros((data.shape[0], 2, data.shape[1]))
    data_complex[:, 0, :] = np.real(data[:,:])  # 每个样本按列分成I路
    data_complex[:, 1, :] = np.imag(data[:,:])  # 每个样本按列分成Q路
    return data_complex

def convert_to_I_Q_complex(data):
    '''Convert the loaded data to complex I and Q samples.'''
    num_row = data.shape[0]#第0维度的大小
    num_col = data.shape[1]#第1维度的大小
    data_complex = np.zeros([num_row, 2, round(num_col/2)])
    data_complex[:,0,:] = data[:,:round(num_col/2)]#每个样本按列分成I路
    data_complex[:,1,:] = data[:,round(num_col/2):]#每个样本按列分成Q路

    return data_complex


def convert_to_I_Q_complex_augment(data):
    '''Convert the loaded data to complex I and Q samples.'''
    num_row = data.shape[0]  # 第0维度的大小
    num_col = data.shape[1]  # 第1维度的大小
    data_complex = np.zeros(([num_row, round(num_col / 2)]), dtype=np.complex64)
    data_complex[:, :] = data[:, :round(num_col / 2)]+1j*data[:, round(num_col / 2):]  # 每个样本按列分成I路
    # data_complex[:, 1, :] = data[:, round(num_col / 2):]  # 每个样本按列分成Q路

    return data_complex


def LoadDataset(file_path, dev_range, pkt_range):
    '''
    Load IQ sample from a dataset
    Input:
    file_path is the dataset path
    dev_range specifies the loaded device range
    pkt_range specifies the loaded packets range

    Return:
    data is the loaded complex IQ samples
    label is the true label of each received packet
    '''

    dataset_name = 'data'
    labelset_name = 'label'

    f = h5py.File(file_path, 'r')
    label = f[labelset_name][:]#获取文件中名字为labelset_name的dataset对应的值
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1
    #label是按照设备index从小到大排列的，所以要+1为设备号
    label_start = int(label[0]) + 1
    label_end = int(label[-1]) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt/num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ',' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()#从index=0开始，找到label里面和设备index相同的都找出来
        sample_index_list.extend(sample_index_dev)
    #从设备index=0开始到dev_range-1，每个设备pkt_range个数据样本按照顺序取出来
    data = f[dataset_name][sample_index_list]
    data = convert_to_I_Q_complex(data)
    label = label[sample_index_list]
    # with open("IQdata/IQdata.txt", 'w') as train_los:
    #     train_los.write(str(data[0,0,:]))

    f.close()
    return data, label


def LoadDataset_augment(file_path, dev_range, pkt_range):
    dataset_name = 'data'
    labelset_name = 'label'

    f = h5py.File(file_path, 'r')
    label = f[labelset_name][:]  # 获取文件中名字为labelset_name的dataset对应的值
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1
    # label是按照设备index从小到大排列的，所以要+1为设备号
    label_start = int(label[0]) + 1
    label_end = int(label[-1]) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt / num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ',' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()  # 从index=0开始，找到label里面和设备index相同的都找出来
        sample_index_list.extend(sample_index_dev)

    K=2048#截取IQ数据长度
    L=11#多径信道长度
    aum_size=100#多每个样本扩增数量
    # 从设备index=0开始到dev_range-1，每个设备pkt_range个数据样本按照顺序取出来
    data = f[dataset_name][sample_index_list]
    data = convert_to_I_Q_complex_augment(data)[:, 0:K]#转换成复数
    label = label[sample_index_list]

    complex_numbers=np.empty((aum_size, L), dtype=np.complex64)
    convolved_result = np.empty((data.shape[0]*aum_size, data.shape[1]+L-1), dtype=np.complex64)
    label_convolved= np.empty(data.shape[0]*aum_size, dtype=np.int8)
    for k in range(0, aum_size):
        temp=np.random.randn(L) + 1j * np.random.randn(L)
        complex_numbers[k,:] = temp

    for i in range(len(data)):
        # complex_data = data[i, 0, :] + 1j * data[i, 1, :]
        for k in range(0, aum_size):
            # complex_numbers = np.random.randn(L) + 1j * np.random.randn(L)
            convolved_result[aum_size*i+k] = np.convolve(data[i,:], complex_numbers[k,:])
            label_convolved[aum_size*i+k] = label[i]

    data_after_convolved=complex_convert_to_I_Q_real(convolved_result)



    f.close()
    return data_after_convolved, label_convolved

#
# def Get_LoRa_ALLIQDataset(file_path, dev_range, pkt_range):  # 获得所有的IQ数据集
#     X, Y = LoadDataset(file_path, dev_range, pkt_range)
#     Y = Y.astype(np.uint8)
#     return X, Y


def Get_LoRa_IQDataset(file_path, dev_range, pkt_range):  # 获得所有的IQ数据集，并将划分为独立的训练集，验证集和测试集
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=30)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def get_num_class_Sourcetraindata(num):  # 从文件中获得num个设备编号从[0,到num-1]对应的设备的源训练和验证集，按顺序
    file_path = 'F:\LoRa_RFFI\dataset\Train\dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    # train_index_shot = []
    # val_index_shot = []
    # for i in range(num):
    #     train_index_shot += [index for index, value in enumerate(Y_train) if value == i]
    #     val_index_shot += [index for index, value in enumerate(Y_val) if value == i]
    # return X_train[train_index_shot], X_val[val_index_shot], Y_train[train_index_shot], Y_val[val_index_shot]
    Y_train = np.squeeze(Y_train)
    Y_val = np.squeeze(Y_val)
    return X_train, X_val, Y_train, Y_val


def get_num_class_Targettraindata(num):  # 从目标文件中获得num个设备编号从[0,到num-1]对应的设备的目标训练和验证集，按顺序
    file_path = 'F:\LoRa_RFFI\dataset\Test\dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_spectrogram_Dataset(file_path, dev_range, pkt_range)
    train_index_shot = []
    val_index_shot = []
    for i in range(num):
        train_index_shot += [index for index, value in enumerate(Y_train) if value == i]
        val_index_shot += [index for index, value in enumerate(Y_val) if value == i]
    return X_train[train_index_shot], X_val[val_index_shot], Y_train[train_index_shot], Y_val[val_index_shot]


def get_num_class_Sourcetestdata(num):  # 从目标文件中获得num个设备编号从[0,到num-1]对应的设备的源测试集，按顺序
    file_path = 'F:\LoRa_RFFI\dataset\Train\dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    test_index_shot = []
    for i in range(num[0], num[1]):
        test_index_shot += [index for index, value in enumerate(Y_train) if value == i]

    Y_train = np.squeeze(Y_train)
    return X_train[test_index_shot], Y_train[test_index_shot]


def get_num_class_Targettestdata(num):  # 从目标文件中获得num个设备编号从[0,到num-1]对应的设备的目标测试集，按顺序
    file_path = 'F:\LoRa_RFFI\dataset\Test\dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    test_index_shot = []
    for i in range(num[0], num[1]):
        test_index_shot += [index for index, value in enumerate(Y_test) if value == i]

    Y_test = np.squeeze(Y_test)
    return X_test[test_index_shot], Y_test[test_index_shot]


def get_num_class_Sourcetrainfinetunedata(num,k):  # 从目标文件中文件中获得num个设备编号从[0,到num-1]对应的设备的源（训练和验证）集，从中每个设备随机取k个数据，按顺序，再分成训练和验证集
    file_path = 'F:\LoRa_RFFI\dataset\Train\dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    train_val_finetune_index_shot = []

    for i in range(num):
        train_index_classi = [index for index, value in enumerate(Y_train_val) if value == i]
        train_val_finetune_index_shot += random.sample(train_index_classi, k)

    X_fintune_train_val, Y_fintune_train_val = X_train_val[train_val_finetune_index_shot], Y_train_val[
        train_val_finetune_index_shot]
    X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val = train_test_split(X_fintune_train_val,
                                                                                      Y_fintune_train_val,
                                                                                      test_size=0.2, random_state=30)

    return X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val


def get_num_class_Targettrainfinetunedata(num,k):  # 从目标文件中文件中获得num个设备编号从[0,到num-1]对应的设备的目标（训练和验证）集，从中每个设备随机取k个数据，按顺序，再分成训练和验证集
    file_path = 'F:\LoRa_RFFI\dataset\Test\dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    train_val_finetune_index_shot = []

    for i in range(num):
        train_index_classi = [index for index, value in enumerate(Y_train_val) if value == i]
        train_val_finetune_index_shot += random.sample(train_index_classi, k)

    X_fintune_train_val, Y_fintune_train_val = X_train_val[train_val_finetune_index_shot], Y_train_val[
        train_val_finetune_index_shot]
    X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val = train_test_split(X_fintune_train_val,
                                                                                      Y_fintune_train_val,
                                                                                      test_size=0.2, random_state=30)
    Y_fintune_train = np.squeeze(Y_fintune_train)
    Y_fintune_val = np.squeeze(Y_fintune_val)
    return X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val


def get_num_class_TargetSemitraindata(num,k):  # 取出num个设备编号从[0,到num-1]对应的设备的目标（训练和测试）集，从中每个设备随机取k个数据，再分成训练和验证集；取出所有剩下的（训练和测试）集
    file_path = 'F:\LoRa_RFFI\dataset\Test\dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    train_val_label_index_shot = []
    train_val_unlabel_index_shot = []

    for i in range(num):
        train_index_classi = [index for index, value in enumerate(Y_train_val) if value == i]
        train_val_label_index_shot += random.sample(train_index_classi, k)

    for k in range(len(X_train_val)):  # 把刚才没选出来的index的，全部选出来放一起
        if k not in train_val_label_index_shot:
            train_val_unlabel_index_shot = np.append(train_val_unlabel_index_shot, k)

    train_val_unlabel_index_shot = train_val_unlabel_index_shot.astype('int64')

    X_label_train_val, Y_label_train_val = X_train_val[train_val_label_index_shot], Y_train_val[
        train_val_label_index_shot]  # 取出num个设备编号从[0,到num-1]对应的设备的目标（训练和测试）集，从中每个设备随机取k个数据，再分成训练和验证集
    X_label_train, X_label_val, Y_label_train, Y_label_val = train_test_split(X_label_train_val, Y_label_train_val,
                                                                              test_size=0.2, random_state=30)  #

    X_unlabel_train_val, Y_unlabel_train_val = X_train_val[train_val_unlabel_index_shot], Y_train_val[
        train_val_unlabel_index_shot]  # 取出所有剩下的（训练和测试）集

    return X_label_train, X_unlabel_train_val, X_label_val, \
        Y_label_train, Y_unlabel_train_val, Y_label_val


def rand_bbox(size, mask_ratio):
    length = size[2]
    cut_length = np.int32(length * mask_ratio)
    cx = np.random.randint(length)  # 0到length-1随机选一个整数作为mask的中点
    bbx1 = np.clip(cx - cut_length // 2, 0, length)  # mask的起点
    bbx2 = np.clip(cx + cut_length // 2, 0, length)  # mask的终点
    return bbx1, bbx2


def MaskData(data, mask_ratio):
    bbx1, bbx2 = rand_bbox(data.size(), mask_ratio)
    data_temp = torch.tensor(data)
    data_temp[:, :, bbx1: bbx2] = torch.zeros(data.size()[1], bbx2 - bbx1).cuda()  # 对所有样本的，每个通道的bbx1到bbx2位置设置0
    return bbx1, bbx2, data_temp


if __name__ == '__main__':
    num = 30
    X_label_train, X_unlabel_train_val, X_label_val, Y_label_train, Y_unlabel_train_val, Y_label_val = get_num_class_TargetSemitraindata(
        num, 20)
    print('success')
