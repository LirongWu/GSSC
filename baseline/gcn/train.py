import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

import csv
import pickle
from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'coauthor-cs':
        data = dgl.data.CoauthorCSDataset()
    elif args.dataset == 'coauthor-phy':
        data = dgl.data.CoauthorPhysicsDataset() 
    else:
        print('Unknown dataset: {}'.format(args.dataset))

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

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(torch.device('cuda'))

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

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    outFile = open('PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in args.__dict__.items():
        results.append(k)
    
    results.append(str(acc))

    time_start = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features)
    time_end = time.time()

    results.append(str(time_end-time_start))
    writer.writerow(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--model_name", type=str, default="gcn")
    parser.add_argument("--mode", type=int, default=0,
                        help="label mode")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
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
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
