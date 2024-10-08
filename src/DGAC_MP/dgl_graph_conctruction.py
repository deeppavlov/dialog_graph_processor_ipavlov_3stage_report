import torch
import numpy as np
import dgl
from torch.utils.data import DataLoader

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_dgl_graphs(x, y, top_k, batch_size, is_no_test = True):
    data = []
    y = torch.tensor(y)

    for i in range(len(x)):
        graph = x[i][0]
        embs = x[i][1]
        
        top_k_nums = list(range(top_k))
        g = dgl.graph((top_k_nums[:-1], top_k_nums[1:]), num_nodes = top_k)
        g.ndata['attr'] = torch.tensor(graph, dtype=torch.int64)
        g.ndata['emb'] = torch.tensor(embs)
        g = dgl.add_self_loop(g)
        data.append((g, y[i]))
    
    data = DataLoader(data, batch_size=batch_size, shuffle=is_no_test, drop_last=False, collate_fn=collate)

    return data