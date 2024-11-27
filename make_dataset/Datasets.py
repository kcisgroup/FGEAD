import torch
from torch.utils.data import Dataset
from config import *
from sklearn.preprocessing import MinMaxScaler


class Datasets(Dataset):
    def __init__(self, raw_data, edge_index, window_size, mode='train'):
        self.raw_data = raw_data

        self.edge_index = edge_index
        self.mode = mode
        self.window_size = window_size

        x_data = raw_data[:-1]
        labels = raw_data[-1]
        # features * numbers
        data = x_data
        # data = self.normalize(data)
        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()
        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        # 窗口大小、滑动距离
        slide_win, slide_stride = self.window_size, SLIDE_STRIDE

        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        for i in rang:
            # 取slide_win大小的特征和标签
            ft = data[:, i - slide_win:i]
            tar = data[:, i]
            # 历史序列封装， 目标序列的封装（features,slide_win）
            x_arr.append(ft)
            y_arr.append(tar)
            # 目标序列的标签
            labels_arr.append(labels[i])
        #  沿一个新维度对输入张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度，而stack（）函数指定的dim参数，就是新增维度的（下标）位置。
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，使得两个tensor完全没有联系，类似于深拷贝
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()
        return feature, y, label, edge_index

    def normalize(self, data):
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        return data





