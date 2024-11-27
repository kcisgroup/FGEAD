from config import *
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as F
from my_models.Encoder import Encoder
from my_models.graph_layers import GraphLayer



def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    # （2， edge_num * batch_num）
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


# self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)
# n个线性映射（batch_size, in_num）->(batch_size, 1)   # (batch_num, node_num, 1)  out = self.out_layer(out)
class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=16):
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                # 对批次内的各个特征进行归一化
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        # nn.ModuleList,它是一个存储不同module，并自动将每个module的parameters添加到网络之中的容器。
        # 你可以把任意nn.Module的子类（如nn.Conv2d，nn.Linear等）加到这个list里面，方法和python自带的list一样，
        # 无非是extend，append等操作，但不同于一般的list，加入到nn.ModuleList里面的module是会自动注册到整个网络上的，
        # 同时module的parameters也会自动添加到整个网络中。若使用python的list，则会出问题。
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                # (batch_size, slide_win, features)
                out = out.permute(0, 2, 1)
                out = mod(out)
                # (batch_size, features, slide_win)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=4, node_num=nsl_kdd_nodes_num):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, alpha) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = alpha
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out), self.att_weight_1


class GAD(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=EmbSize, out_layer_inter_dim=256, input_dim=10, out_layer_num=1,
                 topk=topk):
        super(GAD, self).__init__()
        # 边索引 # 边索引(1, 2, edge_num)
        self.edge_index_sets = edge_index_sets
        # (2, edge_num)
        self.input_size = input_dim
        edge_index = edge_index_sets[0]
        embed_dim = dim
        # (2, edge_num)
        self.encoder = Encoder(node_num, dim)
        # 也就是对批次内的特征进行归一化
        self.bn_outlayer_in = nn.BatchNorm1d(dim)
        # 1
        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.__alpha = None
        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None
        self.dp = nn.Dropout(0.2)


    def forward(self, data, org_edge_index):
        # (batch_size, feature_num, slide_win)
        x = data.clone().detach()
        # (2, edge_num)
        edge_index_sets = self.edge_index_sets
        device = data.device
        batch_num, node_num, all_feature = x.shape
        # (batch_num * node_num, win_length)
        review_x = x.view(-1, all_feature).contiguous()
        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            # （2， batch_size * (node_num-1)）
            cache_edge_index = self.cache_edge_index_sets[i]
            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            batch_edge_index = self.cache_edge_index_sets[i]

            weight_arr, embed_x = self.encoder(torch.arange(node_num).to(device))


            weight_arr = weight_arr.sum(dim=0)
            # (node_num, dim)
            all_embeddings = embed_x.repeat(batch_num, 1)
            topk_num = self.topk
            # (node_num, topk)
            topk_indices_ji = torch.topk(weight_arr, topk_num, dim=-1)[1]
            # topk_values = torch.topk(weight_arr, topk_num, dim=-1)[0]
            self.learned_graph = topk_indices_ji
            # (1, node_num * topk)
            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            # values_gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            # values_gated_j = topk_values.flatten().unsqueeze(0)
            # values_edge_index = torch.cat((gated_j, gated_i), dim=0)


            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out, self.__alpha = self.gnn_layers[i](review_x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings, )
            gcn_outs.append(gcn_out)
        # 所有N个节点的表示
        x = torch.cat(gcn_outs, dim=1)

        # (batch_size, node_num, node_num)
        x = x.view(batch_num, node_num, -1)
        indexes = torch.arange(0, node_num).to(device)
        # （batch_num, node_num, emb_dim）
        out = torch.mul(x, embed_x)
        # (batch_num , emb_dim, node_num)
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        # (batch_num, node_num, emb_dim)
        out = out.permute(0, 2, 1)
        out = self.dp(out)
        # (batch_num, node_num, 1)
        out = self.out_layer(out)
        # (batch_num, node_num)
        out = out.view(-1, node_num)
        return out, embed_x, gated_edge_index, batch_num, weight_arr, self.__alpha

    def compute_loss(self, out, target):
        recon_loss = torch.nn.MSELoss()(out, target)
        return recon_loss

    def compute_batch_error(self, out, target):
        loss = torch.nn.MSELoss(reduction='none')(out, target)
        batch_error = loss.mean(1)
        return batch_error
