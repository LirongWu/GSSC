"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import csv
import pickle

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        data = load_data(args)
    elif args.dataset == 'coauthor-cs':
        data = dgl.data.CoauthorCSDataset()
    elif args.dataset == 'coauthor-phy':
        data = dgl.data.CoauthorPhysicsDataset() 
    if args.dataset == 'film':
        adj = pickle.load(open('../dataset/film/{}_adj.pkl'.format(args.dataset), 'rb'))
        row, col = np.where(adj.todense() > 0)

        U = row.tolist()
        V = col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)

        g.ndata['feat'] = torch.FloatTensor(pickle.load(open('../dataset/film/{}_features.pkl'.format(args.dataset), 'rb')))
        g.ndata['label'] = torch.LongTensor(pickle.load(open('../dataset/film/{}_labels.pkl'.format(args.dataset), 'rb')))
    else:
        g = data[0]

    if args.mode < -20 and args.mode >= -50:
        src = np.load("../dataset/{}/src_mask_r{}.npy".format(args.dataset, -args.mode))
        dst = np.load("../dataset/{}/dst_mask_r{}.npy".format(args.dataset, -args.mode))
        U = src.tolist()
        V = dst.tolist()
        graph = dgl.graph((U, V))

        if g.number_of_nodes() - graph.number_of_nodes() > 0:
            graph.add_nodes(g.number_of_nodes() - graph.number_of_nodes())


        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph)

        graph.ndata['feat'] = g.ndata['feat']
        graph.ndata['label'] = g.ndata['label']
        graph.ndata['train_mask'] = g.ndata['train_mask']
        graph.ndata['val_mask'] = g.ndata['val_mask']
        graph.ndata['test_mask'] = g.ndata['test_mask']

        g = graph

    features = g.ndata['feat']
    labels = g.ndata['label']
    if args.mode < -50 and args.mode >= -60:
        labels = torch.LongTensor(np.load("../dataset/{}/label_mask_r{}.npy".format(args.dataset, -args.mode))).to(torch.device('cuda'))

    if args.dataset == 'cora' or args.dataset == 'citeseer':
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()

        if args.mode < 0 and args.mode >= -20:
            train_index = np.load("../dataset/{}/train_mask_r{}.npy".format(args.dataset, -args.mode))
            val_index = np.load("../dataset/{}/val_mask_r{}.npy".format(args.dataset, -args.mode))
            test_index = np.load("../dataset/{}/test_mask_r{}.npy".format(args.dataset, -args.mode))
            train_mask = torch.zeros(labels.shape[0], dtype=bool)
            val_mask = torch.zeros(labels.shape[0], dtype=bool)
            test_mask = torch.zeros(labels.shape[0], dtype=bool)
            train_mask[train_index] = True
            val_mask[val_index] = True
            test_mask[test_index] = True
    else:
        train_index = np.load("../dataset/{}/train_mask.npy".format(args.dataset))
        val_index = np.load("../dataset/{}/val_mask.npy".format(args.dataset))
        test_index = np.load("../dataset/{}/test_mask.npy".format(args.dataset))
        train_mask = torch.zeros(labels.shape[0], dtype=bool)
        val_mask = torch.zeros(labels.shape[0], dtype=bool)
        test_mask = torch.zeros(labels.shape[0], dtype=bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        in_feats = features.shape[1]
        n_classes = int(labels.max().item() + 1)
        n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_nid)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, g, features, labels, test_nid)
    print("Test Accuracy {:.4f}".format(acc))

    outFile = open('PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in args.__dict__.items():
        results.append(k)
    
    results.append(str(acc))

    time_start = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    time_end = time.time()

    results.append(str(time_end-time_start))
    writer.writerow(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--model_name", type=str, default="graphsage")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--mode", type=int, default=0,
                        help="label mode")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)
