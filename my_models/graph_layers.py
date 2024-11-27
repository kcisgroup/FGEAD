from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from config import *
# gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
# GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1)
# 基于图的注意力机制预测
class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)
        # slide_window
        self.in_channels = in_channels
        # dim
        # self.thea = thea
        self.thea = Parameter(torch.Tensor(1))
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.__alpha__ = None
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        # torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。
        # 即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        #（1， heads, out_channels）  att_l和att_j对应公式中的a^T,l和j也是分别用于source node和target node
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # glorot认为优秀的初始化应该使得各层的激活值和状态梯度的方差在传播过程中的方差保持一致。
        # glorot假设DNN使用激活值关于0对称且在0处梯度为1的激活函数(如tanh)。
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)
    # (batch_num * node_num, slide_win)  （2， node_num * topk_num * batch_size） 批边结构的索引
    # edge_index:[2, num_edges]，在这个包含两行的数组中，第1行与第2行中对应索引位置的值分别表示一条边的源节点和目标节点 # （2， node_num * topk_num * batch_size） 批边结构的索引
    def forward(self, x, edge_index, embedding, return_attention_weights=True):
        """"""
        if torch.is_tensor(x):
            # (batch_num * node_num, out_channels)  #相当于公式中的Theta，此处和接下来的else内容 是因为其GAT实现了Theta1和Theta2分别用于source node和target node
            x = self.lin(x)
            # (2, batch_num * node_num, out_channels)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))
        # contains_self_loops(edge_index)：判断图中节点是否包含自环。
        # remove_self_loops(edge_index)：删除图中所有的自环。
        # add_self_loops(edge_index)：为图中的节点添加自环，对于有自环的节点，它会再为该节点添加一个自环。
        # add_remaining_self_loops：为图中还没有自环的节点添加自环。
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))
        # 进行消息传递
        # 邻居信息变换：\phi
        # 邻居信息聚合：\square
        # 自身信息与聚合后邻居信息的变换得到自身的最终信息：\gamma
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):
        """
        x_j和alpha_j是source node的信息
        index是与source node相连的target node的标号
        alpha_i顺序与edge_index第二行一致
        :param x_i:
        :param x_j:
        :param edge_index_i:
        :param size_i:
        :param embedding:
        :param edges:
        :param return_attention_weights:
        :return:
        """
        # (batch_size, 1, 64)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        # print("x_i:", x_i.shape)
        # (batch_size, 1, 64)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            # print("embedding_i_1:", embedding_i.shape)
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            # print("embedding_i:", embedding_i.shape)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)
            # (batch_size, 1, dim+embed)
            key_i = torch.cat((x_i, self.thea * embedding_i), dim=-1)
            # print("key_i:", key_i.shape)
            key_j = torch.cat((x_j, self.thea * embedding_j), dim=-1)

        # (1, heads, 2 * out_channels)
        cat_att_i = torch.cat((self.att_i,  self.att_em_i), dim=-1)
        # print("cat_att_i:", cat_att_i.shape)
        cat_att_j = torch.cat((self.att_j,  self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print("alpha_5:", alpha.shape)
        # (batch_size, 1, 64)
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
