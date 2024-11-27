import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from torch.autograd import Variable
from my_models.MultiHeadSelfAttention import MultiHeadSelfAttention


class Encoder(nn.Module):
    def __init__(self, node_num, EmbSize):
        super(Encoder, self).__init__()

        self.node_num = node_num
        self.embedding = nn.Embedding(node_num, EmbSize)
        self.multi_attention = MultiHeadSelfAttention(4, EmbSize//4, AttentionDropout)
        self.init_weights()

    def init_weights(self, w=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, inputs):
        x = self.embedding(inputs)
        attention_score, attention_x = self.multi_attention(x)

        return attention_score, attention_x

