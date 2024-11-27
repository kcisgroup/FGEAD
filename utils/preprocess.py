# 字典（顶点-》边），特征， 所有特征， 返回的是边索引
def build_loc_net(struc, all_features, feature_map=[]):
    # 顶点集
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    # node:一个顶点,node_list:其余顶点
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        # 对列表使用 index(a) 是查找列表中的对应元素 a 的第一个位置索引值
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes

# 返回（features_num+1, len_data）
def construct_data(data, feature_map, labels=0):
    res = []
    # 依次获取各个特征的数据
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])
    # 添加标签
    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)
    return res