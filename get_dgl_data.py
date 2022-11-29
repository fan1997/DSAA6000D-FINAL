from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
import numpy as np
from dgl.data import RedditDataset
from torch.utils.data import DataLoader

# file = "arxiv"
# data = DglNodePropPredDataset(name="ogbn-arxiv")
# graph, labels = data[0]

# file = "protein"
# data = DglNodePropPredDataset(name="ogbn-proteins")
# graph, labels = data[0]

# file = "products"
# data = DglNodePropPredDataset(name="ogbn-products")
# graph, labels = data[0]

# file = "ddi"
# data = DglLinkPropPredDataset(name="ogbl-ddi")
# print(data)
# graph = data[0]

# file = "ppa"
# data = DglLinkPropPredDataset(name="ogbl-ppa")
# print(data)
# graph = data[0]

# file = "collab"
# data = DglLinkPropPredDataset(name="ogbl-collab")
# print(data)
# graph = data[0] 

file = "biokg"
data = DglLinkPropPredDataset(name="ogbl-biokg")
print(data)
graph = data[0]

# file = "reddit"
# data = RedditDataset()
# graph = data[0]


# add reverse edges
srcs, dsts = graph.all_edges()
graph.add_edges(dsts, srcs)
graph = graph.remove_self_loop().add_self_loop()
srcs, dsts = graph.all_edges()
print(srcs.shape)
print(dsts.shape)
np.savez(file + '.npz', src_li = srcs, dst_li = dsts, num_nodes = graph.number_of_nodes())





# file = "molhiv"
# data = DglGraphPropPredDataset(name="ogbg-molhiv")
# file = "code2"
# data = DglGraphPropPredDataset(name="ogbg-code2")
# split_idx = data.get_idx_split()
# train_loader = DataLoader(data[split_idx["train"]],
#                               batch_size=8096,
#                               shuffle=True,
#                               collate_fn=collate_dgl)
# for graph, _ in train_loader:
#     srcs, dsts = graph.all_edges()
#     graph.add_edges(dsts, srcs)
#     graph = graph.remove_self_loop().add_self_loop()
#     srcs, dsts = graph.all_edges()
#     print(srcs.shape)
#     print(dsts.shape)
#     np.savez(file + '.npz', src_li = srcs, dst_li = dsts, num_nodes = graph.number_of_nodes())
#     exit(0)


