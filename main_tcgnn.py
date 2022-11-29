import os.path as osp
import argparse
import os
import sys
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *
import TCGNN
from dataset import *
from gnn_conv import *
from config import *
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--num_layers", type=int, default=2, help="num layers")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=10, help="number of epoches")
parser.add_argument("--model", type=str, default='gcn_b', help='GNN model', choices=['gcn_b', 'gcn', 'gin', 'agnn'])
parser.add_argument("--single_kernel", action='store_true', help="whether to profile a single SAG kernel")
parser.add_argument("--single_kernel_b", action='store_true', help="whether to profile a single SAG balance kernel")
parser.add_argument("--single_kernel_b_v1", action='store_true', help="whether to profile a single SAG balance kernel")
parser.add_argument("--single_kernel_t", action='store_true', help="whether to test a single SAG balance kernel")
args = parser.parse_args()
print(args)
#########################################
## Load Graph from files.
#########################################
dataset = args.dataset
path = osp.join("tcgnn-ae-graphs/", dataset + ".npz")
dataset = TCGNN_dataset(path, args.dim, args.classes, load_from_txt=False)
num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index =  dataset.column_index 
row_pointers = dataset.row_pointers
#########################################
## Compute TC-GNN related graph MetaData.
#########################################
num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
edgeToRow = torch.zeros(num_edges, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
# preprocessing for generating meta-information
start = time.perf_counter()
# ADD FAN:
TCblockRowid, TCblockColid, TCblocktileRowid, TCblocktileColid, TCblockoffset, SparseAToXindex, block_count = TCGNN.preprocess_v1(column_index, row_pointers, num_nodes, BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)
build_neighbor_parts = time.perf_counter() - start
print("Prep. (ms):\t{:.3f}".format(build_neighbor_parts*1e3))
# ADD FAN:
print("blockPartition:", blockPartition)
np.savetxt(args.dataset + ".csv", blockPartition.numpy(), delimiter=",") 
##FAN: analysis the nnz in each TC block
num_nnz_per_block = torch.zeros(block_count, dtype=torch.int)
for i in range(block_count):
    num_nnz_per_block[i] = TCblockoffset[i+1] - TCblockoffset[i] 
print("mean nnz in TC tile:", torch.mean(num_nnz_per_block.float()))
print("nnz >= 16 in TC tile:", num_nnz_per_block[num_nnz_per_block >= 16].shape)
print("nnz >= 12 in TC tile:", num_nnz_per_block[num_nnz_per_block >= 12].shape)

#########################################
## ADD FAN: Create COO format for TC blocks.
#########################################
column_index = column_index.cuda()
row_pointers = row_pointers.cuda()
blockPartition = blockPartition.cuda()
edgeToColumn = edgeToColumn.cuda()
edgeToRow = edgeToRow.cuda()
TCblockColid = TCblockColid.cuda()
TCblockRowid = TCblockRowid.cuda()
TCblocktileRowid = TCblocktileRowid.cuda()
TCblocktileColid = TCblocktileColid.cuda()
TCblockoffset = TCblockoffset.cuda() 
SparseAToXindex = SparseAToXindex.cuda() 
print("TCblockRowid:", TCblockRowid.shape)
print("TCblockColid:", TCblockColid.shape)
print("TCblocktileRowid:", TCblocktileRowid)
print("TCblocktileRowid:", TCblocktileRowid.shape)
print("TCblocktileColid:", TCblocktileColid)
print("TCblocktileColid:", TCblocktileColid.shape)
print("TCblockoffset:", TCblockoffset)
print("TCblockoffset:", TCblockoffset.shape)
print("SparseAToXindex:", SparseAToXindex)
print("SparseAToXindex:", SparseAToXindex.shape)
print("column_index:", column_index)
print("column_index:", column_index.shape)
print("row_pointers:", row_pointers)
print("row_pointers:", row_pointers.shape)
print("blockPartition:", blockPartition)
print("blockPartition:", blockPartition.shape)
print("edgeToColumn:", edgeToColumn)
print("edgeToColumn:", edgeToColumn.shape)
print("edgeToRow:", edgeToRow)
print("edgeToRow:", edgeToRow.shape)

#########################################
## Single Satter-And-Gather (SAG) Profiling.
#########################################
if args.single_kernel:
    SAG_obj = SAG(row_pointers, column_index,\
                    blockPartition, edgeToColumn, edgeToRow)
    X = dataset.x
    SAG_out = SAG_obj.profile(X, 10)
    exit(0)
if args.single_kernel_b:
    SAG_obj = SAG_balance(row_pointers, column_index,\
                    blockPartition, edgeToColumn, edgeToRow, TCblockRowid, TCblockColid, block_count)
    X = dataset.x
    SAG_obj.profile(X, 10)
    exit(0)
if args.single_kernel_b_v1:
    SAGbv1_obj = SAG_balance_v1(TCblockRowid, TCblockColid,\
                    TCblocktileRowid, TCblocktileColid, TCblockoffset, SparseAToXindex, num_nodes)
    X = dataset.x
    SAGbv1_out = SAGbv1_obj.profile(X, 10)
    exit(0)
if args.single_kernel_t:
    SAGbv1_obj = SAG_balance_v1(TCblockRowid, TCblockColid,\
                    TCblocktileRowid, TCblocktileColid, TCblockoffset, SparseAToXindex, num_nodes)
    SAG_obj = SAG(row_pointers, column_index,\
                    blockPartition, edgeToColumn, edgeToRow)
    X = dataset.x
    SAGbv1_out = SAGbv1_obj.profile(X, 1)
    SAG_out = SAG_obj.profile(X, 1)
    print("max absolute error:", torch.max(torch.abs(SAG_out - SAGbv1_out)))
    relerr = (SAG_out - SAGbv1_out) / SAG_out
    relerr[relerr != relerr] = 0.0
    print("max relative error:", torch.max(torch.abs(relerr)))
    exit(0)

#########################################
## Build GCN and AGNN Model
#########################################
if args.model == "gcn":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden)

            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(GCNConv(args.hidden, args.hidden))
            
            self.conv2 = GCNConv(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            for Gconv  in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

if args.model == "gcn_b":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv_Balance(dataset.num_features, args.hidden)

            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(GCNConv_Balance(args.hidden, args.hidden))
            
            self.conv2 = GCNConv_Balance(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, TCblockRowid, TCblockColid, block_count))
            x = F.dropout(x, training=self.training)
            for Gconv  in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, TCblockRowid, TCblockColid, block_count)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, TCblockRowid, TCblockColid, block_count)
            return F.log_softmax(x, dim=1)

elif args.model == "gin":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.conv1 = GINConv(dataset.num_features, args.hidden)
            self.hidden_layers = nn.ModuleList()
            for i in range(args.num_layers -  2):
                self.hidden_layers.append(GINConv(args.hidden, args.hidden))
            self.conv2 = GINConv(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            for Gconv in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

elif args.model == "agnn":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = AGNNConv(dataset.num_features, args.hidden)
            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(AGNNConv(args.hidden, args.hidden))
            self.conv2 = AGNNConv(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            for Gconv in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, dataset = Net().to(device), dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training 
def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[:], dataset.y[:])
    loss.backward()
    print(loss)
    optimizer.step()

# if __name__ == "__main__":
#     # dry run.
#     for epoch in range(1, 3):
#         train()
#     torch.cuda.synchronize()
#     start_train = time.perf_counter()
#     for _ in tqdm(range(1, args.epochs + 1)):
#         train()
#     torch.cuda.synchronize()
#     train_time = time.perf_counter() - start_train

#     print("Train (ms):\t{:6.3f}".format(train_time*1e3/args.epochs))