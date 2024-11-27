# -*- coding: utf-8 -*-
import time
from my_models.GAD import GAD
from datetime import datetime
from utils.visualize import *
from torch.utils.data import DataLoader, Subset
from make_dataset.Datasets import Datasets
from utils.preprocess import *
from torch.optim import lr_scheduler
from config import *
from sklearn.manifold import TSNE
import torch.nn.functional as F



# 设置输出的最大阈值
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(linewidth=50)


def get_feature_map(feature_list_path):
    feature_file = open(feature_list_path, 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list


#  一个字典，顶点--边  一个顶点到其余顶点
def get_fc_graph_struc(feature_list_path):
    feature_file = open(feature_list_path, 'r')
    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map


def get_dataset(train_path, test_path, feature_list_path, window_size=3):
    train_orig = pd.read_csv(train_path, sep=',', index_col=0)
    test_orig = pd.read_csv(test_path, sep=',', index_col=0)
    train, test = train_orig, test_orig
    if 'attack' in train.columns:
        train = train.drop(columns=['attack'])

    # 获取所有特征列表  get data
    feature_map = get_feature_map(feature_list_path)
    # 构造一个字典（顶点--边）
    fc_struc = get_fc_graph_struc(feature_list_path)
    # 输入 数据， 特征  # 返回（features_num+1, len_data）
    # 输入顶点--》边， 特征， 所有特征， 返回的是边索引（第一个是起始顶点下标， 第二个是目标顶点下标）
    fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
    # 转换成整形
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    # (features+1, data_length)
    train_dataset_indata = construct_data(train, feature_map, labels=0)
    test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())
    train_dataset = Datasets(train_dataset_indata, fc_edge_index, window_size, mode='train')
    test_dataset = Datasets(test_dataset_indata, fc_edge_index, window_size, mode='test')

    return feature_map, fc_edge_index, train_dataset, test_dataset


def train(model, train_dataloader, epoch,  best_val_loss, optimizer, scheduler, model_file):
    t = time.time()
    total_loss = []
    model.train()
    dataloader = train_dataloader
    print1 = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.float().to(Device) for item in [x, y, labels, edge_index]]
        optimizer.zero_grad()
        if print1 == 0 and epoch == 0:
            # x.shape [32, 125, 5], labels.shape [32, 125], attack_labels.shape [32]
            print('x.shape, y.shape, labels.shape, edge_index.shape', x.shape, y.shape, labels.shape, edge_index.shape)
        y_pred, embed_x, gated_edge_index, batch_num, weight_arr, alpha_ = model(x, edge_index)
        loss = model.compute_loss(y_pred, y)

        total_loss.append(loss.item())
        loss.backward()
        loss = optimizer.step()
        scheduler.step()
        print1 = 1


    print('Epoch: {:04d}'.format(epoch),
          'total_loss: {:.10f}'.format(np.mean(total_loss)),
          'time: {:.4f}s'.format(time.time() - t))

    if np.mean(total_loss) < best_val_loss:
        best_val_loss = np.mean(total_loss)
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')


def run_train(train_path, test_path, feature_list_path, count=0, topk=5, window=7, emb_size=32):
    feature_map, fc_edge_index, train_dataset, test_dataset = get_dataset(train_path, test_path, feature_list_path, window)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)
    edge_index_sets = []
    # 边索引(1, 2, edge_num)
    edge_index_sets.append(fc_edge_index)

    model = GAD(edge_index_sets, len(feature_map), emb_size, out_layer_inter_dim, window, out_layer_num, topk).float().to(Device)
    model.cuda()
    # save model
    # model_file = "./Result/" + str(count) + "best.pt"
    model_file = "./Result/" + "topk_" + str(topk) + "thea_" + str(thea) + "window_" + str(window) + "emb_size_" + str(emb_size) + "nsl_kdd_best.pt"
    best_loss = np.inf
    optimizer = torch.optim.Adam(list(model.parameters()), lr=Lr, weight_decay=decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)
    try:
        for epoch in range(EPOCH):
            train(model, train_dataloader,
                epoch,  best_loss, optimizer, scheduler, model_file)
        # test()
    except KeyboardInterrupt:
        pass

    start_time = datetime.now()
    test_labels, error = test(model, test_dataloader)
    end_time = datetime.now()
    evaluation_time = (end_time-start_time).seconds
    test_labels = np.array(test_labels.cpu().numpy())
    print(
        'thea:{}'.format(thea),
        'topk:{}'.format(topk),
        'window:{}'.format(window),
        'emb_size:{}'.format(emb_size),
        'evaluation_time:{}'.format(evaluation_time)
          )
    make_figure(test_labels, error)



def test(model, dataloader):
    # test
    now = time.time()

    test_len = len(dataloader)
    model.eval()

    error = []
    test_labels = []
    print1 = 0

    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.float().to(Device) for item in [x, y, labels, edge_index]]
        with torch.no_grad():
            if print1 == 0:
                # x.shape [32, 125, 5], y.shape [32, 125], labels.shape [32]
                print('x.shape, y.shape, labels.shape, edge_index.shape', x.shape, y.shape, labels.shape, edge_index.shape)
            y_pred, embed_x, gated_edge_index, batch_num, weight_arr, alpha_ = model(x, edge_index)

            batch_error = model.compute_batch_error(y_pred, y)

            error += batch_error.detach().tolist()

            if len(test_labels) <= 0:
                test_labels = labels

            else:
                test_labels = torch.cat((test_labels, labels), dim=0)
            print1 = 1

    return test_labels, error




if __name__ == '__main__':
    starttime = datetime.now()
    train_path = r"D:\papers\first_class\GAD\data\UNSW_NB15\train.csv"
    test_path = r"D:\papers\first_class\GAD\data\UNSW_NB15\test.csv"
    feature_list_path = r"D:\papers\first_class\GAD\data\UNSW_NB15\list.txt"
    run_train(train_path, test_path, feature_list_path, 0, 7, 7, 64)
    endtime = datetime.now()
    t = (endtime - starttime).seconds
    print('*************************The total time is ', t)


