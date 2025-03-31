import json
from collections import OrderedDict

# 用于存储节点和其对应id的字典
node_to_id = {}
# 用于给节点分配id的计数器
id_counter = 0

date_file_path = 'D:/lunwen/DyGCN Graphic/data/High-energy physics theory citation network/Cit-HepTh-dates.txt'

# 打开EasyCit.txt文件
with open(date_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 提取每行的第一个元素作为节点
        node = line.strip().split()[0]
        if node not in node_to_id:
            node_to_id[id_counter] = node
            id_counter += 1

# 使用OrderedDict来保持节点顺序与首次出现顺序一致（如果需要的话）
# ordered_node_to_id = OrderedDict(sorted(node_to_id.items(), key=lambda t: t[1]))

# 将结果写入node2id.json文件
with open('D:/lunwen/DyGCN Code/DyGCN-master/data/data_hep/node2id.json', 'w', encoding='utf-8') as json_file:
    json.dump(node_to_id, json_file, ensure_ascii=False, indent=4)