import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec
from tqdm import tqdm


def get_randomwalk(G, node, path_length):
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
    return random_walk


# 基础路径
path_init_file = 'D:/论文/DyGCN Code/DyGCN-master/data/data_hep/'
embedding_path = 'D:/论文/DyGCN Code/DyGCN-master/data/data_hep_emb/'

gamma = 10
walk_length = 5

model = Word2Vec(size=256,
                 window=4,
                 sg=1,
                 hs=0,
                 negative=10,
                 alpha=0.03,
                 min_alpha=0.0007,
                 seed=14
                 )

for present_graph in range(87, 115):
    try:
        # 加载图数据
        path_now = path_init_file + "month_" + str(present_graph) + "_graph.gpickle"
        G = nx.read_gpickle(path_now)

        # 清空 random_walks 列表
        random_walks = []

        all_nodes = list(G.nodes())
        print(all_nodes)

        for n in tqdm(all_nodes):
            for i in range(gamma):
                random_walk = get_randomwalk(G, n, walk_length)
                # 确保随机游走路径中的节点为字符串类型
                random_walk = [str(node) for node in random_walk]
                random_walks.append(random_walk)

        if present_graph == 87:
            # 首次构建词汇表
            model.build_vocab(random_walks, progress_per=2)
        else:
            # 后续更新词汇表
            model.build_vocab(random_walks, update=True, progress_per=2)

        model.train(random_walks, total_examples=model.corpus_count, epochs=50, report_delay=1)

        # 获取节点嵌入向量
        node_embeddings = {}
        for node in G.nodes():
            node_embeddings[str(node)] = model.wv[str(node)]

        # 将嵌入向量转换为 NumPy 数组
        embedding_array = np.array([node_embeddings[str(node)] for node in sorted(G.nodes())])

        # 保存嵌入结果为 .npy 文件
        file_path = embedding_path + 'month_' + str(present_graph) + '_graph_embedding.npy'
        np.save(file_path, embedding_array)

        print(f"嵌入结果已保存到 {file_path}")
    except Exception as e:
        print(f"处理图 {present_graph} 时出现错误: {e}")
