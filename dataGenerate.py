import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random

# 提示开始读取数据
print("开始读取日期文件...")
date_file_path = 'D:/lunwen/DyGCN Graphic/data/High-energy physics theory citation network/Cit-HepTh-dates.txt'
line_file_path = 'D:/lunwen/DyGCN Graphic/data/High-energy physics theory citation network/Cit-HepTh.txt'

# 读取数据到 DataFrame
df = pd.read_csv(date_file_path, sep='\t', header=None, names=['paper_id', 'date'])
# 提示日期文件读取完成
print("日期文件读取完成。")

# 提示开始将日期转换为 datetime 类型
print("开始将日期转换为 datetime 类型...")
# 将日期转换为 datetime 类型
df['date'] = pd.to_datetime(df['date'])
# 提示日期转换完成
print("日期转换完成。")

# 提示开始设置索引
print("开始设置索引...")
df = df.set_index('paper_id')
# 提示索引设置完成
print("索引设置完成。")

# 提示开始获取最早和最晚日期
print("开始获取最早和最晚日期...")
# 获取最早和最晚日期
min_date = df['date'].min()
max_date = df['date'].max()
# 提示获取完成
print("最早和最晚日期获取完成。")

# 提示开始生成每隔 3 个月的截止日期
print("开始生成每隔 3 个月的截止日期...")
# 生成每隔 3 个月的截止日期
end_dates = pd.date_range(start=min_date, end=max_date, freq='3M')
total_dates = len(end_dates)
# 提示截止日期生成完成
print("截止日期生成完成。")

# 提示开始读取边文件
print("开始读取边文件...")
dl = pd.read_csv(line_file_path, sep='\t', header=None, names=['node1', 'node2'])
# 提示边文件读取完成
print("边文件读取完成。")

# 提示开始提前构建节点日期字典
print("开始提前构建节点日期字典...")
# 提前构建节点日期字典
node_date_dict = {}
for node, date in df.iterrows():
    if isinstance(date['date'], pd.Series):
        if len(date['date']) > 1:
            print(f"Warning: 节点 {node} 有多个对应日期. 现在使用第一个.")
        node_date_dict[node] = date['date'].iloc[0]
    else:
        node_date_dict[node] = date['date']
# 提示节点日期字典构建完成
print("节点日期字典构建完成。")


def process_date(i, end_date):
    print(f"Processing date {i}/{total_dates}")
    # 提示开始构建图
    print(f"开始为第 {i} 个日期构建图...")
    # 构建图
    G = nx.Graph()
    for _, row in dl.iterrows():
        node1, node2 = row['node1'], row['node2']
        try:
            date1 = node_date_dict[node1]
            date2 = node_date_dict[node2]
            if date1 <= end_date and date2 <= end_date:
                G.add_edge(node1, node2)
        except KeyError:
            # 若 node1 或 node2 不在 df 中，跳过该边
            continue
    # 提示图构建完成
    print(f"第 {i} 个日期的图构建完成。")

    # 为每条边添加 valid 字段，根据随机数划分训练集和验证集
    for u, v in G.edges():
        if random.random() < 0.6:
            G[u][v]['valid'] = False  # 训练集
        else:
            G[u][v]['valid'] = True   # 验证集

    graph_file_path = f'D:/lunwen/DyGCN Code/DyGCN-master/data/data_hep/month_{i + 74}_graph.gpickle'
    # plt.figure(figsize=(14, 14))
    # nx.draw(G)
    # plt.show()
    print(G.number_of_nodes())
    # 提示开始保存图
    print(f"开始保存第 {i} 个日期的图...")
    try:
        nx.write_gpickle(G, graph_file_path)
        print(f"Graph {i + 74} has been saved as {graph_file_path}")
        # 提示图保存成功
        print(f"第 {i} 个日期的图保存成功。")
    except Exception as e:
        print(f"Error saving graph {i + 74}: {e}")
        # 提示图保存失败
        print(f"第 {i} 个日期的图保存失败。")


# 提示开始使用线程池并行处理
print("开始使用线程池并行处理...")
# 使用线程池并行处理
with ThreadPoolExecutor() as executor:
    futures = []
    for i, end_date in enumerate(end_dates[1:], start=1):
        future = executor.submit(process_date, i, end_date)
        futures.append(future)

    # 使用 tqdm 显示进度条
    for future in tqdm(futures, desc="Processing dates", total=len(futures)):
        future.result()
# 提示所有任务完成
print("所有任务完成。")