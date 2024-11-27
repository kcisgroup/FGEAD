import torch
import torch.nn as nn

nsl_kdd_nodes_num = 114
ALL_Dropout = 0.2
# data-preprocess
seed = 5  # 随机种子
BATCH_SIZE = 64  # 批量大小
SLIDE_WIN = 5  # 窗口大小 运行predictor时候使用
SLIDE_STRIDE = 1 # 窗口步长

# self-attention
EmbSize = 32
# Attention参数,AttentionHeadNum能被EmbSize整除
# AttentionHeadNum = 4
# AttentionHeadSize = EmbSize // AttentionHeadNum  # 除法以后是float型，//则是整型
AttentionDropout = ALL_Dropout
# transformer中FeedForward参数
# FFInputSize = EmbSize  # 输出仍然为FFInputSize
# FFIntermediateSize = EmbSize * 2
FFDrop = ALL_Dropout

flag = 0
topk = 7
out_layer_num = 2
out_layer_inter_dim = 16

thea = 0

decay = 0
EPOCH = 30  # 迭代次数
optimizer = 'Adam'  # 优化器
Lr = 4e-4  # 学习率
no_factor = False  # 禁用因素图模型

lr_decay = 200  # After how epochs to decay LR by a factor of gamma
gamma = 1.0  # LR decay factor

N_constraint = 50
Shrink_thres = 1.0/N_constraint


# 显卡
device_num = 0  # 使用第几块显卡

def get_device(device_num):
    cuda_condition = torch.cuda.is_available()
    cuda0 = torch.device('cuda:0' if cuda_condition else 'cpu')
    cuda1 = torch.device('cuda:1' if cuda_condition else 'cpu')
    device = cuda0 if device_num == 0 else cuda1
    print("当前使用的显卡为：", device)
    return device

Device = get_device(device_num)

