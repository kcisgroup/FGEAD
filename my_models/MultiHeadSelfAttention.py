import math
from config import *

# 输入维度：(batch_size,seq_len,emb_size)
# 输出维度：(batch_size,sql_len,emb_size)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 dropout_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size  # 一般注意力输入输出前后维度不变

        # 网络
        self.q_dense = nn.Linear(self.out_dim, self.out_dim)
        self.k_dense = nn.Linear(self.out_dim, self.out_dim)
        self.v_dense = nn.Linear(self.out_dim, self.out_dim)
        self.softmax = nn.Softmax(dim=-1)  # 在第几维上进行softmax运算，这里是最后一维。
        self.dropout = nn.Dropout(dropout_prob)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x):
        # (node_num, emdeding_size)
        qx = x
        kx = x
        vx = x
        # (node_num, Embsize)
        q = self.q_dense(qx)
        k = self.k_dense(kx)
        v = self.v_dense(vx)

        # 先将node_num*embedding_size变成node_num*head*head_size
        # 再将node_num*head*head_size转置成head*node_num*head_size
        shape = list(x.size())
        node_num = shape[0]
        embeding_size = shape[1]
        q = q.view([node_num, self.attention_head_num, self.attention_head_size])
        q = q.transpose(0, 1)
        k = k.view([node_num, self.attention_head_num, self.attention_head_size])
        k = k.transpose(0, 1)
        v = v.view([node_num, self.attention_head_num, self.attention_head_size])
        v = v.transpose(0, 1)
        # q和k的转置相乘得到：[head,node_num,node_num]
        attention_scores = torch.matmul(q, k.transpose(1, 2))
        # 因为q,k相乘，结果变大，对结果除以根号
        attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))

        attention_scores = self.softmax(attention_scores)
        # [head,node_num,node_num]
        attention_scores = self.dropout(attention_scores)
        # [head, node_num, head_size]
        attention_x = torch.matmul(attention_scores, v)
        # permute改变维度，congiguous返回内存中连续的Tensor
        attention_x = attention_x.permute(1, 0, 2).contiguous()
        attention_x = attention_x.view([node_num, self.out_dim])
        attention_x = self.o_dense(attention_x)
        return attention_scores.view([self.attention_head_num, node_num, node_num]), attention_x
