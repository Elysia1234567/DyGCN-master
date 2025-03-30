import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib
print(matplotlib.__version__)

graph_file_path = 'D:/论文/DyGCN Code/DyGCN-master/data/data_hep/month_108_graph.gpickle'
G = nx.read_gpickle(graph_file_path)
plt.figure(figsize=(14,14))
nx.draw(G)
plt.show()